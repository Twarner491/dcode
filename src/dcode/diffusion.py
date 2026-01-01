"""Latent-to-Gcode Diffusion Model.

Architecture:
- Frozen SD VAE encoder: image → latent (4x64x64)
- Trainable transformer decoder: latent → gcode tokens
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
from diffusers import AutoencoderKL


class LatentGcodeConfig(PretrainedConfig):
    model_type = "latent_gcode"
    
    def __init__(
        self,
        latent_dim: int = 4,
        latent_size: int = 64,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        vocab_size: int = 32128,  # T5 tokenizer size
        max_seq_len: int = 4096,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout


class LatentProjector(nn.Module):
    """Project SD latent (4x64x64) to sequence of embeddings."""
    
    def __init__(self, config: LatentGcodeConfig):
        super().__init__()
        self.config = config
        
        # Flatten and project latent
        latent_flat_dim = config.latent_dim * config.latent_size * config.latent_size
        
        # Use a small CNN to reduce spatial dimensions, then project
        self.conv = nn.Sequential(
            nn.Conv2d(config.latent_dim, 64, 3, stride=2, padding=1),  # 64x32x32
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128x16x16
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 256x8x8
            nn.GELU(),
        )
        
        # 256 * 8 * 8 = 16384 → project to sequence
        self.proj = nn.Linear(256 * 8 * 8, config.hidden_size * 64)  # 64 "visual tokens"
        self.num_visual_tokens = 64
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, 4, 64, 64) SD latent
        Returns:
            (B, 64, hidden_size) sequence of visual embeddings
        """
        B = latent.shape[0]
        x = self.conv(latent)  # (B, 256, 8, 8)
        x = x.view(B, -1)  # (B, 16384)
        x = self.proj(x)  # (B, hidden_size * 64)
        x = x.view(B, self.num_visual_tokens, self.config.hidden_size)
        return x


class GcodeTransformerDecoder(nn.Module):
    """Autoregressive transformer decoder for gcode generation."""
    
    def __init__(self, config: LatentGcodeConfig):
        super().__init__()
        self.config = config
        
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_size)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(
        self, 
        visual_embeds: torch.Tensor, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_embeds: (B, num_visual_tokens, hidden_size) from latent projector
            input_ids: (B, seq_len) gcode token ids
            attention_mask: (B, seq_len) attention mask
        Returns:
            logits: (B, seq_len, vocab_size)
        """
        B, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token + position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        token_embeds = self.token_embed(input_ids) + self.pos_embed(positions)
        
        # Causal mask for autoregressive decoding
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        
        # Decode: visual_embeds as memory, token_embeds as target
        hidden = self.decoder(
            tgt=token_embeds,
            memory=visual_embeds,
            tgt_mask=causal_mask,
            tgt_is_causal=True,
        )
        
        hidden = self.ln_f(hidden)
        logits = self.head(hidden)
        
        return logits


class LatentGcodeModel(PreTrainedModel):
    """Full model: SD VAE encoder (frozen) + Gcode decoder (trainable)."""
    
    config_class = LatentGcodeConfig
    
    def __init__(self, config: LatentGcodeConfig):
        super().__init__(config)
        self.config = config
        
        self.projector = LatentProjector(config)
        self.decoder = GcodeTransformerDecoder(config)
        
        # VAE will be loaded separately
        self.vae = None
        
    def load_vae(self, vae_id: str = "stabilityai/sd-vae-ft-mse"):
        """Load frozen SD VAE encoder."""
        self.vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
        self.vae.requires_grad_(False)
        self.vae.eval()
        
    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to SD latents.
        
        Args:
            images: (B, 3, H, W) normalized to [-1, 1]
        Returns:
            latent: (B, 4, H/8, W/8)
        """
        if self.vae is None:
            raise ValueError("VAE not loaded. Call load_vae() first.")
        
        # Ensure images are right size (512x512 for SD)
        if images.shape[-1] != 512:
            images = torch.nn.functional.interpolate(images, size=(512, 512), mode="bilinear")
        
        latent = self.vae.encode(images.half()).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        return latent.float()
    
    def forward(
        self,
        images: torch.Tensor = None,
        latents: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        """
        Args:
            images: (B, 3, H, W) input images (will be encoded to latents)
            latents: (B, 4, 64, 64) pre-computed latents (alternative to images)
            input_ids: (B, seq_len) gcode token ids
            labels: (B, seq_len) target token ids for loss computation
            attention_mask: (B, seq_len)
        """
        # Get latents from images if not provided
        if latents is None:
            if images is None:
                raise ValueError("Either images or latents must be provided")
            latents = self.encode_image(images)
        
        # Project latents to visual embeddings
        visual_embeds = self.projector(latents)
        
        # Decode to gcode logits
        logits = self.decoder(visual_embeds, input_ids, attention_mask)
        
        loss = None
        if labels is not None:
            # Shift for autoregressive loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        return {"loss": loss, "logits": logits}
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor = None,
        latents: torch.Tensor = None,
        tokenizer = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> list[str]:
        """Generate gcode from images or latents."""
        self.eval()
        device = next(self.parameters()).device
        
        if latents is None:
            latents = self.encode_image(images.to(device))
        
        visual_embeds = self.projector(latents)
        B = visual_embeds.shape[0]
        
        # Start with BOS token
        bos_id = tokenizer.bos_token_id or tokenizer.pad_token_id or 0
        input_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        
        eos_id = tokenizer.eos_token_id or 1
        
        for _ in range(max_new_tokens):
            logits = self.decoder(visual_embeds, input_ids)
            next_logits = logits[:, -1, :] / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            
            for b in range(B):
                next_logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = float('-inf')
            
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for EOS
            if (next_token == eos_id).all():
                break
        
        # Decode tokens to text
        outputs = []
        for ids in input_ids:
            text = tokenizer.decode(ids, skip_special_tokens=True)
            outputs.append(text)
        
        return outputs


def create_model_and_tokenizer(config: LatentGcodeConfig = None):
    """Create model and tokenizer for training."""
    if config is None:
        config = LatentGcodeConfig()
    
    model = LatentGcodeModel(config)
    model.load_vae()
    
    # Use T5 tokenizer - good for code-like text
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    
    return model, tokenizer

