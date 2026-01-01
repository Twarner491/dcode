"""Corrected SD-Gcode training.

Architecture:
- SD (frozen): text → text_encoder → UNet → latent
- Decoder (trained): latent → gcode

Training:
- Use VAE-encoded IMAGE latents (deterministic)
- NOT random diffusion latents

Inference:
- text → frozen SD → latent → trained decoder → gcode

Output: Single model file containing all weights.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


class GcodeDecoderConfig:
    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: int = 64,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        vocab_size: int = 32128,
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
    def __init__(self, config: GcodeDecoderConfig):
        super().__init__()
        self.config = config
        
        self.latent_proj = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size * 16),
            nn.LayerNorm(config.hidden_size * 16),
        )
        
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_size)
        
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
        
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        
    def forward(self, latent: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        latent_flat = latent.view(batch_size, -1)
        memory = self.latent_proj(latent_flat)
        memory = memory.view(batch_size, 16, self.config.hidden_size)
        
        positions = torch.arange(seq_len, device=device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        
        x = self.decoder(x, memory, tgt_mask=causal_mask)
        x = self.ln_f(x)
        return self.lm_head(x)


class PreEncodedDataset(Dataset):
    """Dataset with pre-encoded latents for fast training."""
    
    def __init__(self, latents: torch.Tensor, gcode_ids: torch.Tensor):
        self.latents = latents
        self.gcode_ids = gcode_ids
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        return {
            "latent": self.latents[idx],
            "gcode_ids": self.gcode_ids[idx],
        }


def _load_and_transform(args):
    """Load image and read gcode (I/O bound, parallel)."""
    sample, transform = args
    try:
        img = Image.open(sample["image"]).convert("RGB")
        img_tensor = transform(img)
        gcode = sample["gcode"].read_text(errors="ignore")[:20000]
        return img_tensor, gcode, True
    except Exception:
        return None, None, False


def _tokenize(args):
    """Tokenize gcode (CPU bound, parallel)."""
    gcode, tokenizer, max_len = args
    tokens = tokenizer.encode(
        gcode,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).squeeze(0)
    return tokens


def pre_encode_dataset(
    manifest_path: Path,
    vae: AutoencoderKL,
    tokenizer,
    max_gcode_len: int,
    cache_dir: Path,
    batch_size: int = 256,
    num_workers: int = 64,
):
    """Pre-encode all images to latents and tokenize gcode. Cache to disk."""
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    import multiprocessing as mp
    
    cache_file = cache_dir / "latents_v2.pt"
    
    if cache_file.exists():
        print(f"Loading cached latents from {cache_file}...")
        data = torch.load(cache_file)
        return PreEncodedDataset(data["latents"], data["gcode_ids"])
    
    print("Pre-encoding dataset (one-time, fully parallel)...")
    
    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    if isinstance(manifest, list):
        pairs = manifest
    elif "pairs" in manifest:
        pairs = manifest["pairs"]
    else:
        pairs = list(manifest.values())[0] if manifest else []
    
    # Filter valid samples
    samples = []
    for item in pairs:
        img_path = Path(item["image"])
        gcode_path = Path(item["gcode"])
        if img_path.exists() and gcode_path.exists():
            samples.append({"image": img_path, "gcode": gcode_path})
    
    print(f"Found {len(samples)} valid samples")
    print(f"Using {num_workers} I/O workers, batch size {batch_size}")
    
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])
    
    all_latents = []
    all_gcode_ids = []
    
    # Use thread pool for I/O (loading images/gcode from disk)
    io_executor = ThreadPoolExecutor(max_workers=num_workers)
    
    # Pre-submit next batch while current batch encodes
    pending_batch = None
    batch_iter = range(0, len(samples), batch_size)
    
    for i in tqdm(batch_iter, desc="Encoding"):
        batch_samples = samples[i:i+batch_size]
        
        # If we have a pending batch from prefetch, use it
        if pending_batch is not None:
            images, gcodes = pending_batch
            pending_batch = None
        else:
            # Load this batch
            args = [(s, transform) for s in batch_samples]
            results = list(io_executor.map(_load_and_transform, args))
            images = [r[0] for r in results if r[2]]
            gcodes = [r[1] for r in results if r[2]]
        
        if not images:
            continue
        
        # Start loading next batch in background
        next_i = i + batch_size
        if next_i < len(samples):
            next_batch = samples[next_i:next_i+batch_size]
            next_args = [(s, transform) for s in next_batch]
            future_results = io_executor.map(_load_and_transform, next_args)
        
        # Encode current batch on GPU
        img_batch = torch.stack(images).to(vae.device, vae.dtype)
        with torch.no_grad():
            latents = vae.encode(img_batch).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            latents_cpu = latents.cpu()
        
        # Tokenize in parallel (use threads since tokenizer releases GIL)
        token_args = [(g, tokenizer, max_gcode_len) for g in gcodes]
        tokens_list = list(io_executor.map(_tokenize, token_args))
        
        # Collect results
        for k in range(len(gcodes)):
            all_latents.append(latents_cpu[k])
            all_gcode_ids.append(tokens_list[k])
        
        # Get prefetched next batch
        if next_i < len(samples):
            next_results = list(future_results)
            pending_batch = (
                [r[0] for r in next_results if r[2]],
                [r[1] for r in next_results if r[2]]
            )
    
    io_executor.shutdown()
    
    # Stack
    all_latents = torch.stack(all_latents)
    all_gcode_ids = torch.stack(all_gcode_ids)
    
    print(f"Encoded {len(all_latents)} samples")
    print(f"Latents: {all_latents.shape}, Gcode: {all_gcode_ids.shape}")
    
    # Cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": all_latents, "gcode_ids": all_gcode_ids}, cache_file)
    print(f"Cached to {cache_file}")
    
    return PreEncodedDataset(all_latents, all_gcode_ids)


def train(
    manifest_path: str,
    output_dir: str = "checkpoints/sd_gcode_v2",
    sd_model_id: str = "runwayml/stable-diffusion-v1-5",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    max_gcode_len: int = 1024,
    gradient_accumulation: int = 4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    print(f"Batch size: {batch_size}, Grad accum: {gradient_accumulation}, Effective: {batch_size * gradient_accumulation}")
    
    # Load VAE for encoding images (this is the ONLY SD component we need for training)
    print(f"Loading VAE from {sd_model_id}...")
    vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", torch_dtype=torch.float16)
    vae = vae.to(device)
    vae.eval()
    vae.requires_grad_(False)  # Freeze VAE
    
    # Tokenizer for gcode
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    
    # Pre-encode dataset (fast training)
    cache_dir = Path(output_dir) / "cache"
    dataset = pre_encode_dataset(
        Path(manifest_path),
        vae,
        tokenizer,
        max_gcode_len,
        cache_dir,
        batch_size=256,  # Large batches for GPU
        num_workers=64,  # Parallel I/O
    )
    
    if len(dataset) == 0:
        raise ValueError("No valid samples found in dataset!")
    
    # Free VAE memory after encoding
    del vae
    torch.cuda.empty_cache()
    print("Freed VAE memory")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Safe now - no CUDA in workers
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    
    # Create decoder (this is what we train)
    config = GcodeDecoderConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=max_gcode_len,
    )
    decoder = GcodeDecoder(config).to(device)
    decoder.train()
    
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder params: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Optimizer - ONLY decoder (SD is frozen)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)
    
    # Training loop
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(epochs):
        decoder.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            latent = batch["latent"].to(device, dtype=torch.float16)
            gcode_ids = batch["gcode_ids"].to(device)
            
            # Skip empty batches
            if latent.abs().sum() == 0:
                continue
            
            # Teacher forcing
            decoder_input = gcode_ids[:, :-1]
            decoder_target = gcode_ids[:, 1:]
            
            # Forward
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = decoder(latent, decoder_input)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    decoder_target.reshape(-1),
                    ignore_index=tokenizer.pad_token_id,
                )
                loss = loss / gradient_accumulation
            
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item() * gradient_accumulation
            pbar.set_postfix(loss=f"{loss.item() * gradient_accumulation:.4f}", step=global_step)
            
            # Save checkpoint
            if global_step % 1000 == 0 and global_step > 0:
                ckpt_path = output_path / f"checkpoint-{global_step}"
                ckpt_path.mkdir(exist_ok=True)
                torch.save(decoder.state_dict(), ckpt_path / "decoder.pt")
                print(f"\nSaved checkpoint to {ckpt_path}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    # Save final - COMPLETE model (SD + trained decoder) as single checkpoint
    final_path = output_path / "final"
    final_path.mkdir(exist_ok=True)
    
    print("Packaging complete model (SD + trained decoder)...")
    
    # Load full SD pipeline for saving (pretrained weights since SD is frozen)
    pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
    
    # Build complete state dict with all weights
    complete_state = {}
    
    # Add SD text encoder weights (pretrained, frozen)
    for name, param in pipe.text_encoder.named_parameters():
        complete_state[f"text_encoder.{name}"] = param.data.cpu()
    print(f"Added {sum(1 for k in complete_state if k.startswith('text_encoder'))} text_encoder weights")
    
    # Add SD UNet weights (pretrained, frozen)
    for name, param in pipe.unet.named_parameters():
        complete_state[f"unet.{name}"] = param.data.cpu()
    print(f"Added {sum(1 for k in complete_state if k.startswith('unet'))} UNet weights")
    
    # Add TRAINED decoder weights
    for name, param in decoder.named_parameters():
        complete_state[f"gcode_decoder.{name}"] = param.data.cpu()
    print(f"Added {sum(1 for k in complete_state if k.startswith('gcode_decoder'))} decoder weights (TRAINED)")
    
    # Save complete model
    torch.save(complete_state, final_path / "pytorch_model.bin")
    print(f"Saved {len(complete_state)} total weight tensors")
    
    # Save config
    with open(final_path / "config.json", "w") as f:
        json.dump({
            "sd_model_id": sd_model_id,
            "num_inference_steps": 20,
            "gcode_decoder": {
                "latent_channels": config.latent_channels,
                "latent_size": config.latent_size,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "vocab_size": config.vocab_size,
                "max_seq_len": config.max_seq_len,
            }
        }, f, indent=2)
    
    # Save tokenizers
    tokenizer.save_pretrained(final_path / "gcode_tokenizer")
    pipe.tokenizer.save_pretrained(final_path / "clip_tokenizer")
    
    del pipe
    torch.cuda.empty_cache()
    
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {final_path}")
    print(f"Contains: frozen SD (text_encoder + UNet) + trained gcode_decoder")
    print(f"{'='*50}")
    
    return str(final_path)


if __name__ == "__main__":
    import sys
    manifest = sys.argv[1] if len(sys.argv) > 1 else "data/processed/captioned.json"
    train(manifest)
