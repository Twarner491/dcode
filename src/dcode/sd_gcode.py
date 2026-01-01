"""Stable Diffusion post-trained for text→gcode generation.

Single model: text → SD text encoder → SD UNet (diffusion) → gcode decoder → gcode
All weights trained end-to-end.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import Optional
import math


class GcodeDecoderConfig:
    """Configuration for gcode decoder."""
    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: int = 64,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        vocab_size: int = 32128,  # T5 tokenizer vocab
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.latent_dim = latent_channels * latent_size * latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout


class GcodeDecoder(nn.Module):
    """Transformer decoder: SD latent → gcode tokens."""
    
    def __init__(self, config: GcodeDecoderConfig):
        super().__init__()
        self.config = config
        
        # Project flattened latent to sequence of hidden states
        self.latent_proj = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size * 16),  # 16 memory tokens
            nn.LayerNorm(config.hidden_size * 16),
        )
        
        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_size)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.num_layers)
        
        # Output head
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embed.weight
        
    def forward(
        self,
        latent: torch.Tensor,  # [batch, 4, 64, 64]
        input_ids: torch.Tensor,  # [batch, seq_len]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Project latent to memory sequence
        latent_flat = latent.view(batch_size, -1)  # [batch, 4*64*64]
        memory = self.latent_proj(latent_flat)  # [batch, hidden*16]
        memory = memory.view(batch_size, 16, self.config.hidden_size)  # [batch, 16, hidden]
        
        # Token + position embeddings
        positions = torch.arange(seq_len, device=device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        
        # Causal mask for autoregressive decoding
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        
        # Decode
        x = self.decoder(x, memory, tgt_mask=causal_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        latent: torch.Tensor,
        tokenizer,
        max_length: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> str:
        """Autoregressive generation from latent."""
        device = latent.device
        batch_size = latent.shape[0]
        
        # Start with BOS token
        input_ids = torch.full((batch_size, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            logits = self(latent, input_ids)
            next_logits = logits[:, -1, :] / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            
            for b in range(batch_size):
                next_logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = float('-inf')
            
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        return tokenizer.decode(input_ids[0], skip_special_tokens=True)


class SDGcodeModel(nn.Module):
    """Stable Diffusion post-trained for text→gcode.
    
    Single forward pass: text → CLIP → UNet diffusion → latent → gcode decoder → gcode
    """
    
    def __init__(
        self,
        sd_model_id: str = "stabilityai/stable-diffusion-2-1-base",
        gcode_config: Optional[GcodeDecoderConfig] = None,
        num_inference_steps: int = 20,
    ):
        super().__init__()
        self.sd_model_id = sd_model_id
        self.num_inference_steps = num_inference_steps
        
        # Load SD components via diffusers pipeline (handles auth properly)
        print(f"Loading SD components from {sd_model_id}...")
        from diffusers import StableDiffusionPipeline
        
        pipe = StableDiffusionPipeline.from_pretrained(
            sd_model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        
        # Free the pipeline (we extracted what we need)
        del pipe
        
        # Gcode decoder (replaces VAE decoder)
        if gcode_config is None:
            gcode_config = GcodeDecoderConfig()
        self.gcode_decoder = GcodeDecoder(gcode_config)
        
        # Gcode tokenizer
        self.gcode_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        
        print("SD-Gcode model initialized")
    
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        """Encode text prompts to CLIP embeddings."""
        inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        
        encoder_output = self.text_encoder(**inputs)
        return encoder_output.last_hidden_state
    
    def diffuse(
        self,
        text_embeds: torch.Tensor,
        num_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Run diffusion process to generate latent from text embeddings."""
        num_steps = num_steps or self.num_inference_steps
        batch_size = text_embeds.shape[0]
        device = text_embeds.device
        dtype = text_embeds.dtype
        
        # Start from noise
        latent_shape = (batch_size, 4, 64, 64)
        latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
        
        # Set timesteps
        self.scheduler.set_timesteps(num_steps, device=device)
        latents = latents * self.scheduler.init_noise_sigma
        
        # Diffusion loop
        for t in self.scheduler.timesteps:
            latent_input = self.scheduler.scale_model_input(latents, t)
            
            # Predict noise
            noise_pred = self.unet(
                latent_input,
                t,
                encoder_hidden_states=text_embeds,
            ).sample
            
            # Step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        return latents
    
    def forward(
        self,
        prompts: list[str],
        target_ids: Optional[torch.Tensor] = None,
        num_diffusion_steps: int = 20,
    ) -> dict:
        """Full forward pass: text → latent → gcode.
        
        Args:
            prompts: List of text prompts
            target_ids: Target gcode token ids [batch, seq_len] for training
            num_diffusion_steps: Number of diffusion steps
            
        Returns:
            Dict with 'logits' and optionally 'loss'
        """
        # Text → embeddings
        text_embeds = self.encode_text(prompts)
        
        # Embeddings → latent via diffusion
        latents = self.diffuse(text_embeds, num_steps=num_diffusion_steps)
        
        # Latent → gcode logits
        if target_ids is not None:
            # Training: teacher forcing
            decoder_input = target_ids[:, :-1]
            decoder_target = target_ids[:, 1:]
            
            logits = self.gcode_decoder(latents, decoder_input)
            
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                decoder_target.reshape(-1),
                ignore_index=self.gcode_tokenizer.pad_token_id,
            )
            
            return {"logits": logits, "loss": loss, "latents": latents}
        else:
            # Inference: autoregressive generation
            gcode = self.gcode_decoder.generate(
                latents,
                self.gcode_tokenizer,
            )
            return {"gcode": gcode, "latents": latents}
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        num_diffusion_steps: int = 20,
        max_gcode_length: int = 512,
        temperature: float = 0.8,
    ) -> str:
        """Generate gcode from text prompt."""
        text_embeds = self.encode_text([prompt])
        latents = self.diffuse(text_embeds, num_steps=num_diffusion_steps)
        gcode = self.gcode_decoder.generate(
            latents,
            self.gcode_tokenizer,
            max_length=max_gcode_length,
            temperature=temperature,
        )
        return gcode
    
    def save_pretrained(self, path: str):
        """Save complete model."""
        import os
        import json
        os.makedirs(path, exist_ok=True)
        
        # Save config
        config = {
            "sd_model_id": self.sd_model_id,
            "num_inference_steps": self.num_inference_steps,
            "gcode_decoder": {
                "latent_channels": self.gcode_decoder.config.latent_channels,
                "latent_size": self.gcode_decoder.config.latent_size,
                "hidden_size": self.gcode_decoder.config.hidden_size,
                "num_layers": self.gcode_decoder.config.num_layers,
                "num_heads": self.gcode_decoder.config.num_heads,
                "vocab_size": self.gcode_decoder.config.vocab_size,
                "max_seq_len": self.gcode_decoder.config.max_seq_len,
            }
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Save all weights as single checkpoint
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        
        # Save tokenizers
        self.tokenizer.save_pretrained(os.path.join(path, "clip_tokenizer"))
        self.gcode_tokenizer.save_pretrained(os.path.join(path, "gcode_tokenizer"))
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda"):
        """Load complete model."""
        import json
        
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        
        gcode_config = GcodeDecoderConfig(**config["gcode_decoder"])
        model = cls(
            sd_model_id=config["sd_model_id"],
            gcode_config=gcode_config,
            num_inference_steps=config["num_inference_steps"],
        )
        
        state_dict = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=device)
        model.load_state_dict(state_dict)
        
        return model.to(device)

