"""Training script for SD-Gcode: post-train Stable Diffusion for text→gcode.

Single model trained end-to-end: text → SD → gcode
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import os

from .sd_gcode import SDGcodeModel, GcodeDecoderConfig


class TextGcodeDataset(Dataset):
    """Dataset of (caption, gcode) pairs."""
    
    def __init__(
        self,
        manifest_path: Path,
        gcode_tokenizer,
        max_gcode_len: int = 1024,
        max_samples: int = None,
    ):
        self.gcode_tokenizer = gcode_tokenizer
        self.max_gcode_len = max_gcode_len
        
        # Load manifest
        with open(manifest_path) as f:
            data = json.load(f)
        
        pairs = data.get("pairs", data)
        
        # Filter to existing files
        self.samples = []
        for pair in pairs:
            gcode_path = Path(pair.get("gcode", ""))
            caption = pair.get("caption", "")
            if gcode_path.exists() and caption:
                self.samples.append({
                    "caption": caption,
                    "gcode_path": str(gcode_path),
                })
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} text-gcode pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load gcode (truncated to save memory)
        try:
            with open(sample["gcode_path"], 'r', errors='ignore') as f:
                gcode = f.read(100000)  # First 100k chars
        except:
            gcode = "G0 X0 Y0"
        
        # Tokenize gcode
        tokens = self.gcode_tokenizer(
            gcode,
            max_length=self.max_gcode_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "caption": sample["caption"],
            "gcode_ids": tokens["input_ids"].squeeze(0),
        }


def collate_fn(batch):
    """Custom collate to handle text prompts."""
    return {
        "prompts": [b["caption"] for b in batch],
        "gcode_ids": torch.stack([b["gcode_ids"] for b in batch]),
    }


def train(
    manifest_path: str,
    output_dir: str = "checkpoints/sd_gcode",
    sd_model: str = "stabilityai/stable-diffusion-2-1-base",
    epochs: int = 10,
    batch_size: int = 2,
    gradient_accumulation: int = 16,
    learning_rate: float = 1e-5,
    max_gcode_len: int = 512,
    num_diffusion_steps: int = 10,  # Fewer steps during training for speed
    warmup_ratio: float = 0.1,
    save_every: int = 1000,
    seed: int = 42,
):
    """Post-train Stable Diffusion for text→gcode generation."""
    
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    print(f"Batch size: {batch_size}, Grad accum: {gradient_accumulation}, Effective: {batch_size * gradient_accumulation}")
    
    # Create model
    gcode_config = GcodeDecoderConfig(
        hidden_size=768,
        num_layers=6,
        num_heads=12,
        max_seq_len=max_gcode_len,
    )
    
    model = SDGcodeModel(
        sd_model_id=sd_model,
        gcode_config=gcode_config,
        num_inference_steps=num_diffusion_steps,
    )
    model.to(device)
    
    # Enable gradient checkpointing for memory efficiency
    model.unet.enable_gradient_checkpointing()
    model.text_encoder.gradient_checkpointing_enable()
    
    # Convert to bfloat16 for speed (works with GradScaler)
    model.to(torch.bfloat16)
    
    # Dataset
    dataset = TextGcodeDataset(
        Path(manifest_path),
        model.gcode_tokenizer,
        max_gcode_len=max_gcode_len,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    
    # Optimizer - train EVERYTHING
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(dataloader) * epochs // gradient_accumulation
    warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Mixed precision - use bfloat16 (no scaler needed)
    # scaler = GradScaler()  # Not needed for bfloat16
    
    # Training
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    global_step = 0
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            prompts = batch["prompts"]
            gcode_ids = batch["gcode_ids"].to(device)
            
            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(
                    prompts=prompts,
                    target_ids=gcode_ids,
                    num_diffusion_steps=num_diffusion_steps,
                )
                loss = outputs["loss"] / gradient_accumulation
            
            loss.backward()
            
            if (step + 1) % gradient_accumulation == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Save checkpoint
                if global_step % save_every == 0:
                    ckpt_path = output_path / f"checkpoint-{global_step}"
                    model.save_pretrained(str(ckpt_path))
                    print(f"\nSaved checkpoint to {ckpt_path}")
            
            epoch_loss += loss.item() * gradient_accumulation
            num_batches += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item() * gradient_accumulation:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "gpu": f"{torch.cuda.memory_allocated() / 1e9:.1f}GB",
            })
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        
        # Save epoch checkpoint
        ckpt_path = output_path / f"epoch-{epoch+1}"
        model.save_pretrained(str(ckpt_path))
    
    # Save final model
    final_path = output_path / "final"
    model.save_pretrained(str(final_path))
    print(f"\nTraining complete! Model saved to {final_path}")
    
    return str(final_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", "-m", required=True)
    parser.add_argument("--output", "-o", default="checkpoints/sd_gcode")
    parser.add_argument("--sd-model", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--batch-size", "-b", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train(
        manifest_path=args.manifest,
        output_dir=args.output,
        sd_model=args.sd_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accum,
        learning_rate=args.lr,
        max_gcode_len=args.max_len,
        num_diffusion_steps=args.diffusion_steps,
        seed=args.seed,
    )

