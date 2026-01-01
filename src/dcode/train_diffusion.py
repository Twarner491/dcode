"""Training script for Latent-to-Gcode diffusion model - H100 optimized."""

import json
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torchvision import transforms
from tqdm import tqdm
import argparse

from .diffusion import LatentGcodeModel, LatentGcodeConfig


def _read_gcode_file(gcode_path: str, max_chars: int = 50000) -> str:
    """Read a single gcode file, truncated to save memory."""
    try:
        with open(gcode_path, 'r', errors='ignore') as f:
            return f.read(max_chars)  # Only read first 50k chars (~1000 lines)
    except:
        return "G0 X0 Y0"


class PreEncodedDataset(Dataset):
    """Dataset with pre-encoded latents for faster training."""
    
    def __init__(
        self, 
        manifest_path: Path,
        tokenizer,
        vae,
        device: str,
        max_gcode_len: int = 2048,
        image_size: int = 512,
        cache_dir: Path = None,
    ):
        self.tokenizer = tokenizer
        self.max_gcode_len = max_gcode_len
        self.device = device
        self.vae = vae
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Load manifest
        with open(manifest_path) as f:
            data = json.load(f)
        
        pairs = data.get("pairs", data)
        
        # Filter to existing files - use processed images, not source
        self.valid_pairs = []
        for pair in pairs:
            img_path = Path(pair.get("image", ""))  # Use processed image, not source
            gcode_path = Path(pair.get("gcode", ""))
            if img_path.exists() and gcode_path.exists():
                self.valid_pairs.append({
                    "image": str(img_path),
                    "gcode": str(gcode_path),
                })
        
        print(f"Found {len(self.valid_pairs)} valid image-gcode pairs")
        
        # Pre-encode latents if cache doesn't exist
        self.cache_dir = cache_dir or Path("data/latent_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._precompute_latents()
        self._precompute_tokens()
    
    @torch.no_grad()
    def _precompute_latents(self):
        """Pre-encode all images to latents using DataLoader for max throughput."""
        cache_file = self.cache_dir / "latents.pt"
        
        if cache_file.exists():
            print("Loading cached latents...")
            self.latent_tensor = torch.load(cache_file, map_location="cpu", weights_only=True)
            self.latents = None  # Use tensor directly
            print(f"Loaded {len(self.latent_tensor)} cached latents")
            return
        
        print("Pre-encoding images to latents...")
        
        # Create a simple dataset for image loading
        class ImageOnlyDataset(Dataset):
            def __init__(ds_self, pairs, transform):
                ds_self.pairs = pairs
                ds_self.transform = transform
            
            def __len__(ds_self):
                return len(ds_self.pairs)
            
            def __getitem__(ds_self, idx):
                try:
                    img = Image.open(ds_self.pairs[idx]["image"]).convert("RGB")
                    return ds_self.transform(img), idx
                except:
                    return torch.zeros(3, 512, 512), idx
        
        img_dataset = ImageOnlyDataset(self.valid_pairs, self.transform)
        img_loader = DataLoader(
            img_dataset,
            batch_size=128,  # Large batch
            shuffle=False,
            num_workers=24,  # Max parallelism
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )
        
        self.vae.to(self.device)
        self.vae.eval()
        
        # Pre-allocate latent storage
        self.latents = [None] * len(self.valid_pairs)
        
        for images, indices in tqdm(img_loader, desc="Encoding"):
            images = images.to(self.device, dtype=torch.float16, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                latent = self.vae.encode(images).latent_dist.sample()
                latent = latent * self.vae.config.scaling_factor
            
            latent_cpu = latent.float().cpu()
            for i, idx in enumerate(indices.tolist()):
                self.latents[idx] = latent_cpu[i]
        
        # Save cache - stack into single tensor for efficiency
        print("Saving latent cache...")
        latent_tensor = torch.stack(self.latents)
        torch.save(latent_tensor, cache_file)
        print(f"Cached {len(self.latents)} latents to {cache_file} ({latent_tensor.nbytes / 1e9:.1f} GB)")
    
    def __len__(self):
        if hasattr(self, 'latent_tensor') and self.latent_tensor is not None:
            return len(self.latent_tensor)
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        # Get pre-encoded latent
        if self.latents is not None:
            latent = self.latents[idx]
        else:
            latent = self.latent_tensor[idx]
        
        # Get cached tokens
        input_ids = self.token_cache[idx]
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            "latent": latent,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
    def _precompute_tokens(self):
        """Pre-tokenize gcode files in streaming batches to avoid OOM."""
        from concurrent.futures import ThreadPoolExecutor
        
        token_cache_file = self.cache_dir / "tokens.pt"
        
        if token_cache_file.exists():
            print("Loading cached tokens...")
            self.token_cache = torch.load(token_cache_file, map_location="cpu", weights_only=True)
            print(f"Loaded {len(self.token_cache)} cached tokens")
            return
        
        print("Pre-tokenizing gcode (streaming batches)...")
        
        gcode_paths = [p["gcode"] for p in self.valid_pairs]
        batch_size = 100  # Smaller batches to control memory
        n_workers = 16
        
        # Pre-allocate token tensor
        self.token_cache = torch.zeros(
            len(gcode_paths), self.max_gcode_len, dtype=torch.long
        )
        
        for batch_start in tqdm(range(0, len(gcode_paths), batch_size), desc="Tokenizing"):
            batch_end = min(batch_start + batch_size, len(gcode_paths))
            batch_paths = gcode_paths[batch_start:batch_end]
            
            # Read batch in parallel (truncated to 50k chars)
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                gcodes = list(executor.map(_read_gcode_file, batch_paths))
            
            # Tokenize batch
            tokens = self.tokenizer(
                gcodes,
                max_length=self.max_gcode_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            
            self.token_cache[batch_start:batch_end] = tokens["input_ids"]
            
            # Free memory
            del gcodes, tokens
        
        torch.save(self.token_cache, token_cache_file)
        print(f"Cached {len(self.token_cache)} tokens")


def train(
    manifest_path: str,
    output_dir: str = "checkpoints/latent_gcode",
    epochs: int = 10,
    batch_size: int = 32,  # Increased for H100
    learning_rate: float = 1e-4,
    warmup_steps: int = 500,
    gradient_accumulation: int = 1,
    max_gcode_len: int = 2048,
    save_steps: int = 1000,
    device: str = None,
):
    """Train the latent-to-gcode model with H100 optimizations."""
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Training on {device}")
    print(f"Batch size: {batch_size}, Grad accum: {gradient_accumulation}, Effective: {batch_size * gradient_accumulation}")
    
    # Create model - smaller to fit in memory
    config = LatentGcodeConfig(
        max_seq_len=max_gcode_len,
        hidden_size=512,  # Smaller for memory
        num_layers=6,
        num_heads=8,
    )
    model = LatentGcodeModel(config)
    model.load_vae()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    
    # Pre-encode dataset
    dataset = PreEncodedDataset(
        Path(manifest_path),
        tokenizer,
        model.vae,
        device,
        max_gcode_len=max_gcode_len,
    )
    
    # Move model to device (VAE no longer needed after pre-encoding)
    model.vae = None  # Free VAE memory
    torch.cuda.empty_cache()
    
    model.to(device)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing to debug
        pin_memory=True,
    )
    
    # Optimizer - only train projector and decoder
    trainable_params = list(model.projector.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, fused=True)
    
    total_steps = len(dataloader) * epochs // gradient_accumulation
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Mixed precision
    scaler = GradScaler('cuda')
    
    # Training loop
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    global_step = 0
    model.train()
    
    # Skip torch.compile - can hang on first run
    # if hasattr(torch, 'compile'):
    #     print("Compiling model with torch.compile...")
    #     model = torch.compile(model, mode="reduce-overhead")
    
    for epoch in range(epochs):
        epoch_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, batch in enumerate(progress):
            latents = batch["latent"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            
            # Forward pass with AMP
            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(
                    latents=latents,
                    input_ids=input_ids,
                    labels=input_ids,
                    attention_mask=attention_mask,
                )
                loss = outputs["loss"] / gradient_accumulation
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            
            epoch_loss += loss.item() * gradient_accumulation
            
            if (step + 1) % gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                
                progress.set_postfix(
                    loss=f"{loss.item() * gradient_accumulation:.4f}", 
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    gpu=f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
                )
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    save_checkpoint(model, config, tokenizer, optimizer, scheduler, global_step, output_path)
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        
        # Save end of epoch
        save_checkpoint(model, config, tokenizer, optimizer, scheduler, global_step, output_path, is_final=False)
    
    # Save final model
    save_checkpoint(model, config, tokenizer, optimizer, scheduler, global_step, output_path, is_final=True)
    
    print(f"Training complete! Model saved to {output_path}/final")
    return str(output_path / "final")


def save_checkpoint(model, config, tokenizer, optimizer, scheduler, global_step, output_path, is_final=False):
    """Save model checkpoint."""
    if is_final:
        ckpt_path = output_path / "final"
    else:
        ckpt_path = output_path / f"checkpoint-{global_step}"
    
    ckpt_path.mkdir(exist_ok=True)
    
    # Handle compiled model
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    config.save_pretrained(ckpt_path)
    torch.save({
        "projector": model_to_save.projector.state_dict(),
        "decoder": model_to_save.decoder.state_dict(),
    }, ckpt_path / "pytorch_model.bin")
    
    if not is_final:
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
        }, ckpt_path / "training_state.pt")
    
    tokenizer.save_pretrained(ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Train latent-to-gcode model")
    parser.add_argument("-m", "--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("-o", "--output", default="checkpoints/latent_gcode", help="Output directory")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=1)
    
    args = parser.parse_args()
    
    train(
        manifest_path=args.manifest,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_gcode_len=args.max_len,
        gradient_accumulation=args.grad_accum,
    )


if __name__ == "__main__":
    main()
