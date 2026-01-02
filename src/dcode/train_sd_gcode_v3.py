"""SD-Gcode v3: Comprehensive text-to-gcode diffusion training.

Key improvements over v2:
1. Custom gcode tokenizer preserving newlines and structure
2. Larger decoder (12 layers, 1024 hidden, 16 heads = ~200M params)
3. CNN-based latent projection preserving spatial structure
4. Cosine LR schedule with warmup
5. Multi-GPU support via DDP
6. Proper weight saving (no tied weights issue)
7. Text conditioning alignment - train on text→latent→gcode

Architecture:
- SD (frozen): text → latent via diffusion
- Decoder (trained): latent → gcode tokens

Training approach:
- Use BOTH text-derived latents (from SD) AND image latents (from VAE)
- This aligns the decoder to work with both distributions
"""

import json
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDPMScheduler
from PIL import Image
import torchvision.transforms as T
import math
import re
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================
# CUSTOM GCODE TOKENIZER
# ============================================================================

def build_gcode_tokenizer(gcode_files: list[Path], save_path: Path, vocab_size: int = 8192) -> PreTrainedTokenizerFast:
    """Build a custom tokenizer optimized for gcode.
    
    Key features:
    - Preserves newlines as explicit tokens
    - Handles numeric coordinates efficiently
    - Small vocab for faster training
    """
    save_path = Path(save_path)
    tokenizer_path = save_path / "gcode_tokenizer.json"
    
    if tokenizer_path.exists():
        print(f"Loading cached gcode tokenizer from {tokenizer_path}")
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path),
            pad_token="<pad>",
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
        )
        hf_tokenizer.pad_token = "<pad>"
        hf_tokenizer.pad_token_id = hf_tokenizer.convert_tokens_to_ids("<pad>")
        return hf_tokenizer
    
    print(f"Building custom gcode tokenizer from {len(gcode_files)} files...")
    
    # Sample gcode content
    sample_texts = []
    for i, path in enumerate(tqdm(gcode_files[:1000], desc="Sampling gcode")):
        try:
            text = path.read_text(errors="ignore")[:10000]  # First 10k chars per file
            sample_texts.append(text)
        except Exception:
            continue
    
    # Create BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Pre-tokenizer that splits on whitespace but keeps newlines
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern="\n", behavior="isolated"),  # Keep newlines
        pre_tokenizers.Whitespace(),  # Split on spaces
    ])
    
    # Train
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<newline>"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(sample_texts, trainer=trainer)
    
    # Decoder
    tokenizer.decoder = decoders.BPEDecoder()
    
    # Save
    save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    
    # Wrap as HuggingFace tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
    )
    
    # Ensure special tokens are properly set
    hf_tokenizer.pad_token = "<pad>"
    hf_tokenizer.pad_token_id = hf_tokenizer.convert_tokens_to_ids("<pad>")
    
    # Add newline token
    hf_tokenizer.add_tokens(["<newline>"])
    
    print(f"Tokenizer built: {hf_tokenizer.vocab_size} tokens")
    return hf_tokenizer


# ============================================================================
# LARGER DECODER ARCHITECTURE
# ============================================================================

class GcodeDecoderConfigV3:
    """Larger decoder config ~200M params."""
    
    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: int = 64,
        hidden_size: int = 1024,      # Up from 768
        num_layers: int = 12,          # Up from 6
        num_heads: int = 16,           # Up from 12
        vocab_size: int = 8192,        # Custom gcode tokenizer
        max_seq_len: int = 2048,       # Longer sequences
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ):
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.ffn_mult = ffn_mult


class CNNLatentProjector(nn.Module):
    """CNN-based latent projector preserving spatial structure."""
    
    def __init__(self, config: GcodeDecoderConfigV3):
        super().__init__()
        
        # Progressive downsampling CNN
        # Input: (B, 4, 64, 64)
        self.cnn = nn.Sequential(
            nn.Conv2d(config.latent_channels, 64, 3, stride=2, padding=1),  # -> 64x32x32
            nn.LayerNorm([64, 32, 32]),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # -> 128x16x16
            nn.LayerNorm([128, 16, 16]),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # -> 256x8x8
            nn.LayerNorm([256, 8, 8]),
            nn.GELU(),
            nn.Conv2d(256, config.hidden_size, 3, stride=2, padding=1),  # -> hidden x 4x4
            nn.LayerNorm([config.hidden_size, 4, 4]),
            nn.GELU(),
        )
        
        # 4x4 = 16 spatial positions → 16 memory tokens
        self.num_memory_tokens = 16
        
        # Learnable position embeddings for memory tokens
        self.memory_pos = nn.Parameter(torch.randn(1, self.num_memory_tokens, config.hidden_size) * 0.02)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, 4, 64, 64)
        Returns:
            memory: (B, 16, hidden_size)
        """
        B = latent.shape[0]
        x = self.cnn(latent)  # (B, hidden_size, 4, 4)
        x = x.view(B, x.shape[1], -1).transpose(1, 2)  # (B, 16, hidden_size)
        x = x + self.memory_pos.expand(B, -1, -1)
        return x


class GcodeDecoderV3(nn.Module):
    """Large transformer decoder for gcode generation."""
    
    def __init__(self, config: GcodeDecoderConfigV3):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        
        # CNN latent projector (preserves spatial info)
        self.latent_proj = CNNLatentProjector(config)
        
        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.embed_drop = nn.Dropout(config.dropout)
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_size * config.ffn_mult,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,  # Pre-norm for training stability
            )
            for _ in range(config.num_layers)
        ])
        
        # Output head (NOT tied to embeddings for proper saving)
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
    
    def forward(
        self, 
        latent: torch.Tensor, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            latent: (B, 4, 64, 64) SD latent
            input_ids: (B, seq_len) token IDs
            attention_mask: (B, seq_len) optional padding mask
        Returns:
            logits: (B, seq_len, vocab_size)
        """
        B, seq_len = input_ids.shape
        device = input_ids.device
        dtype = latent.dtype
        
        # Project latent to memory
        memory = self.latent_proj(latent)  # (B, 16, hidden)
        
        # Token embeddings + positions
        positions = torch.arange(seq_len, device=device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_drop(x)
        
        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device, dtype=dtype
        )
        
        # Optional padding mask for memory
        memory_key_padding_mask = None
        
        # Apply decoder layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, memory, causal_mask,
                    use_reentrant=False
                )
            else:
                x = layer(
                    x, memory, 
                    tgt_mask=causal_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# DATASET WITH BOTH IMAGE AND TEXT LATENTS
# ============================================================================

class AlignedLatentDataset(Dataset):
    """Dataset with pre-computed latents for fast training.
    
    Contains both:
    - Image latents (from VAE) - what the model should decode
    - Text-derived latents (from SD diffusion) - what we'll use at inference
    
    Training on both helps align the decoder to work with diffusion outputs.
    """
    
    def __init__(
        self,
        image_latents: torch.Tensor,
        text_latents: Optional[torch.Tensor],
        gcode_ids: torch.Tensor,
        mix_ratio: float = 0.5,  # Fraction of text latents to use
    ):
        self.image_latents = image_latents
        self.text_latents = text_latents
        self.gcode_ids = gcode_ids
        self.mix_ratio = mix_ratio if text_latents is not None else 0.0
        
    def __len__(self):
        return len(self.image_latents)
    
    def __getitem__(self, idx):
        # Randomly choose image or text latent
        if self.text_latents is not None and torch.rand(1).item() < self.mix_ratio:
            latent = self.text_latents[idx]
        else:
            latent = self.image_latents[idx]
        
        return {
            "latent": latent,
            "gcode_ids": self.gcode_ids[idx],
        }


# ============================================================================
# PRE-ENCODING WITH PARALLEL WORKERS
# ============================================================================

def pre_encode_dataset_v3(
    manifest_path: Path,
    vae: AutoencoderKL,
    sd_pipe: Optional[StableDiffusionPipeline],
    tokenizer: PreTrainedTokenizerFast,
    max_gcode_len: int,
    cache_dir: Path,
    batch_size: int = 64,  # Batch for VAE encoding
    num_workers: int = 16,  # Conservative thread count
    generate_text_latents: bool = True,
    rank: int = 0,  # For DDP
) -> AlignedLatentDataset:
    """Pre-encode images (and optionally text) to latents."""
    
    cache_file = cache_dir / f"latents_v3_rank{rank}.pt"
    
    if cache_file.exists():
        if rank == 0:
            print(f"Loading cached data from {cache_file}")
        data = torch.load(cache_file)
        return AlignedLatentDataset(
            data["image_latents"],
            data.get("text_latents"),
            data["gcode_ids"],
        )
    
    if rank == 0:
        print("Pre-encoding dataset...")
    
    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    pairs = manifest.get("pairs", manifest) if isinstance(manifest, dict) else manifest
    
    # Gather valid samples with captions
    samples = []
    for item in pairs:
        img_path = Path(item.get("image", ""))
        gcode_path = Path(item.get("gcode", ""))
        caption = item.get("caption", "")
        
        if img_path.exists() and gcode_path.exists():
            samples.append({
                "image": img_path,
                "gcode": gcode_path,
                "caption": caption or "artwork",
            })
    
    if rank == 0:
        print(f"Found {len(samples)} valid samples")
    
    # Image transform
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])
    
    all_image_latents = []
    all_text_latents = []
    all_gcode_ids = []
    
    # Process in batches with parallel I/O
    device = vae.device
    dtype = vae.dtype
    
    from concurrent.futures import ThreadPoolExecutor
    
    def load_sample(s):
        """Load single sample (runs in thread pool)."""
        try:
            img = Image.open(s["image"]).convert("RGB")
            img_tensor = transform(img)
            gcode = s["gcode"].read_text(errors="ignore")[:50000]
            return img_tensor, gcode, s["caption"], True
        except Exception:
            return None, None, None, False
    
    for i in tqdm(range(0, len(samples), batch_size), desc="Encoding", disable=rank != 0):
        batch_samples = samples[i:i + batch_size]
        
        # Parallel I/O loading
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(load_sample, batch_samples))
        
        imgs = []
        gcodes = []
        captions = []
        valid = []
        
        for img_tensor, gcode, caption, is_valid in results:
            if is_valid:
                imgs.append(img_tensor)
                gcodes.append(gcode)
                captions.append(caption)
            valid.append(is_valid)
        
        if not any(valid):
            continue
        
        imgs = torch.stack(imgs).to(device, dtype)
        
        # Encode images to latents
        with torch.no_grad():
            image_latents = vae.encode(imgs).latent_dist.sample()
            image_latents = image_latents * vae.config.scaling_factor
        
        all_image_latents.append(image_latents.cpu())
        
        # Optionally generate text latents via diffusion
        if generate_text_latents and sd_pipe is not None:
            with torch.no_grad():
                # Use few steps for efficiency
                text_results = sd_pipe(
                    captions,
                    num_inference_steps=10,
                    guidance_scale=7.5,
                    output_type="latent",
                )
                text_latents = text_results.images
            all_text_latents.append(text_latents.cpu())
        
        # Tokenize gcode in parallel
        def tokenize_gcode(gcode):
            gcode_clean = gcode.replace("\n", " <newline> ")
            return tokenizer.encode(
                gcode_clean,
                max_length=max_gcode_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).squeeze(0)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            tokens_list = list(executor.map(tokenize_gcode, gcodes))
        
        all_gcode_ids.extend(tokens_list)
    
    # Stack all
    all_image_latents = torch.cat(all_image_latents, dim=0)
    all_gcode_ids = torch.stack(all_gcode_ids)
    
    if all_text_latents:
        all_text_latents = torch.cat(all_text_latents, dim=0)
    else:
        all_text_latents = None
    
    if rank == 0:
        print(f"Encoded {len(all_image_latents)} samples")
        print(f"Image latents: {all_image_latents.shape}")
        if all_text_latents is not None:
            print(f"Text latents: {all_text_latents.shape}")
        print(f"Gcode tokens: {all_gcode_ids.shape}")
    
    # Cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "image_latents": all_image_latents,
        "text_latents": all_text_latents,
        "gcode_ids": all_gcode_ids,
    }, cache_file)
    
    return AlignedLatentDataset(all_image_latents, all_text_latents, all_gcode_ids)


# ============================================================================
# LEARNING RATE SCHEDULER
# ============================================================================

def get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps: int, 
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Cosine schedule with linear warmup."""
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# MULTI-GPU TRAINING
# ============================================================================

def setup_ddp(rank: int, world_size: int):
    """Initialize DDP."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def train(
    manifest_path: str,
    output_dir: str = "checkpoints/sd_gcode_v3",
    sd_model_id: str = "runwayml/stable-diffusion-v1-5",
    epochs: int = 20,
    batch_size: int = 32,  # Optimized for H100 80GB
    learning_rate: float = 3e-4,
    max_gcode_len: int = 2048,
    gradient_accumulation: int = 1,  # Less needed with 8 GPUs
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    generate_text_latents: bool = False,  # Set True for better alignment (slower)
    num_gpus: Optional[int] = None,
    use_wandb: bool = True,
    wandb_project: str = "dcode-sd-gcode-v3",
    wandb_run_name: Optional[str] = None,
):
    """Train the gcode decoder with comprehensive improvements."""
    
    # Determine device setup
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    use_ddp = num_gpus > 1
    rank = 0
    world_size = 1
    
    if use_ddp:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = num_gpus
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    effective_batch = batch_size * world_size * gradient_accumulation
    
    if rank == 0:
        print(f"Training on {num_gpus} GPU(s)")
        print(f"Batch size: {batch_size} x {world_size} = {batch_size * world_size} per step")
        print(f"Gradient accumulation: {gradient_accumulation}")
        print(f"Total effective batch: {effective_batch}")
    
    # Initialize wandb (only on rank 0)
    if use_wandb and WANDB_AVAILABLE and rank == 0:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or f"v3_{num_gpus}gpu_bs{effective_batch}",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "effective_batch_size": effective_batch,
                "learning_rate": learning_rate,
                "max_gcode_len": max_gcode_len,
                "gradient_accumulation": gradient_accumulation,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "num_gpus": num_gpus,
                "sd_model_id": sd_model_id,
                "generate_text_latents": generate_text_latents,
            },
        )
        print("Wandb initialized")
    
    output_path = Path(output_dir)
    cache_dir = output_path / "cache"
    
    # ========== Load VAE for image encoding ==========
    if rank == 0:
        print(f"Loading VAE from {sd_model_id}...")
    vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", torch_dtype=torch.float16)
    vae = vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    
    # ========== Optionally load SD for text latents ==========
    sd_pipe = None
    if generate_text_latents and rank == 0:
        print("Loading SD pipeline for text latent generation...")
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            sd_model_id, torch_dtype=torch.float16, safety_checker=None
        ).to(device)
        sd_pipe.set_progress_bar_config(disable=True)
    
    # ========== Build/load custom gcode tokenizer (only on rank 0) ==========
    manifest = json.load(open(manifest_path))
    pairs = manifest.get("pairs", manifest) if isinstance(manifest, dict) else manifest
    gcode_files = [Path(p.get("gcode", "")) for p in pairs if Path(p.get("gcode", "")).exists()]
    
    # Only rank 0 builds tokenizer, others wait
    if rank == 0:
        tokenizer = build_gcode_tokenizer(gcode_files, cache_dir, vocab_size=8192)
    
    if use_ddp:
        dist.barrier()  # Wait for rank 0 to finish
    
    if rank != 0:
        tokenizer = build_gcode_tokenizer(gcode_files, cache_dir, vocab_size=8192)  # Load cached
    
    # ========== Pre-encode dataset (only rank 0, others load from cache) ==========
    if rank == 0:
        dataset = pre_encode_dataset_v3(
            Path(manifest_path),
            vae,
            sd_pipe,
            tokenizer,
            max_gcode_len,
            cache_dir,
            batch_size=64,  # Safe batch for H100
            num_workers=16,  # Conservative threads
            generate_text_latents=generate_text_latents,
            rank=rank,
        )
    
    if use_ddp:
        dist.barrier()  # Wait for rank 0 to finish encoding
    
    if rank != 0:
        # Other ranks just load the cached data
        dataset = pre_encode_dataset_v3(
            Path(manifest_path),
            vae,
            sd_pipe,
            tokenizer,
            max_gcode_len,
            cache_dir,
            batch_size=64,
            num_workers=16,
            generate_text_latents=generate_text_latents,
            rank=rank,
        )
    
    # Free memory
    del vae
    if sd_pipe is not None:
        del sd_pipe
    torch.cuda.empty_cache()
    
    if rank == 0:
        print(f"Dataset size: {len(dataset)}")
    
    # ========== DataLoader ==========
    if use_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=4, pin_memory=True, drop_last=True,
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )
    
    # ========== Create decoder ==========
    config = GcodeDecoderConfigV3(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=max_gcode_len,
    )
    decoder = GcodeDecoderV3(config).to(device)
    decoder.enable_gradient_checkpointing()
    
    if rank == 0:
        num_params = decoder.count_params()
        print(f"Decoder params: {num_params:,} ({num_params/1e6:.1f}M)")
    
    if use_ddp:
        decoder = DDP(decoder, device_ids=[rank])
    
    # ========== Optimizer & Scheduler ==========
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    
    total_steps = len(dataloader) * epochs // gradient_accumulation
    warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    if rank == 0:
        print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
    
    # ========== Training Loop ==========
    output_path.mkdir(parents=True, exist_ok=True)
    global_step = 0
    best_loss = float("inf")
    
    for epoch in range(epochs):
        if use_ddp:
            sampler.set_epoch(epoch)
        
        decoder.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", disable=rank != 0)
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            latent = batch["latent"].to(device, dtype=torch.float16)
            gcode_ids = batch["gcode_ids"].to(device)
            
            # Skip empty
            if latent.abs().sum() == 0:
                continue
            
            # Teacher forcing
            decoder_input = gcode_ids[:, :-1]
            decoder_target = gcode_ids[:, 1:]
            
            # Forward with autocast
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = decoder(latent, decoder_input) if not use_ddp else decoder.module(latent, decoder_input)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    decoder_target.reshape(-1),
                    ignore_index=tokenizer.pad_token_id,
                )
                loss = loss / gradient_accumulation
            
            loss.backward()
            epoch_loss += loss.item() * gradient_accumulation
            num_batches += 1
            
            if (batch_idx + 1) % gradient_accumulation == 0:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                lr = scheduler.get_last_lr()[0]
                step_loss = loss.item() * gradient_accumulation
                pbar.set_postfix(loss=f"{step_loss:.4f}", lr=f"{lr:.2e}", step=global_step)
                
                # Wandb logging (every 10 steps for efficiency)
                if use_wandb and WANDB_AVAILABLE and rank == 0 and global_step % 10 == 0:
                    wandb.log({
                        "train/loss": step_loss,
                        "train/learning_rate": lr,
                        "train/grad_norm": grad_norm.item(),
                        "train/epoch": epoch + batch_idx / len(dataloader),
                        "train/gpu_memory_gb": torch.cuda.memory_allocated() / 1e9,
                    }, step=global_step)
                
                # Checkpoint
                if global_step % 2000 == 0 and rank == 0:
                    ckpt_path = output_path / f"checkpoint-{global_step}"
                    ckpt_path.mkdir(exist_ok=True)
                    torch.save(
                        decoder.state_dict() if not use_ddp else decoder.module.state_dict(),
                        ckpt_path / "decoder.pt"
                    )
                    print(f"\nSaved checkpoint to {ckpt_path}")
        
        avg_loss = epoch_loss / max(num_batches, 1)
        
        if rank == 0:
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
            
            # Wandb epoch logging
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch/loss": avg_loss,
                    "epoch/epoch": epoch + 1,
                }, step=global_step)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    # ========== Save Final Model ==========
    if rank == 0:
        final_path = output_path / "final"
        final_path.mkdir(exist_ok=True)
        
        print("Packaging final model...")
        
        # Load fresh SD for complete checkpoint
        pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
        
        # Build complete state dict
        complete_state = {}
        
        # SD text encoder (frozen)
        for name, param in pipe.text_encoder.named_parameters():
            complete_state[f"text_encoder.{name}"] = param.data.cpu()
        
        # SD UNet (frozen)
        for name, param in pipe.unet.named_parameters():
            complete_state[f"unet.{name}"] = param.data.cpu()
        
        # Trained decoder
        decoder_state = decoder.state_dict() if not use_ddp else decoder.module.state_dict()
        for name, param in decoder_state.items():
            complete_state[f"gcode_decoder.{name}"] = param.cpu()
        
        # Save
        torch.save(complete_state, final_path / "pytorch_model.bin")
        print(f"Saved {len(complete_state)} weight tensors")
        
        # Save decoder config
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
                    "ffn_mult": config.ffn_mult,
                }
            }, f, indent=2)
        
        # Save tokenizer
        tokenizer.save_pretrained(final_path / "gcode_tokenizer")
        pipe.tokenizer.save_pretrained(final_path / "clip_tokenizer")
        
        del pipe
        torch.cuda.empty_cache()
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best loss: {best_loss:.4f}")
        print(f"Model saved to: {final_path}")
        print(f"{'='*60}")
        
        # Final wandb summary
        if use_wandb and WANDB_AVAILABLE:
            wandb.summary["best_loss"] = best_loss
            wandb.summary["total_steps"] = global_step
            wandb.summary["final_model_path"] = str(final_path)
            wandb.finish()
    
    if use_ddp:
        cleanup_ddp()
    
    return str(output_path / "final")


if __name__ == "__main__":
    import sys
    manifest = sys.argv[1] if len(sys.argv) > 1 else "data/processed/captioned.json"
    train(manifest)
