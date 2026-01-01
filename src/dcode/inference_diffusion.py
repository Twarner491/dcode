"""Inference for Latent-to-Gcode model with full Stable Diffusion pipeline."""

import torch
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer
from diffusers import StableDiffusionPipeline
from torchvision import transforms

from .diffusion import LatentGcodeModel, LatentGcodeConfig


class DcodeInference:
    """Text-to-Gcode inference using SD + trained gcode decoder."""
    
    def __init__(
        self,
        gcode_model_path: str,
        sd_model_id: str = "stabilityai/stable-diffusion-2-1-base",
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading models on {self.device}...")
        
        # Load Stable Diffusion pipeline
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            sd_model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(self.device)
        
        # Load gcode model
        model_path = Path(gcode_model_path)
        config = LatentGcodeConfig.from_pretrained(model_path)
        
        self.gcode_model = LatentGcodeModel(config)
        
        # Load weights
        weights = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
        self.gcode_model.projector.load_state_dict(weights["projector"])
        self.gcode_model.decoder.load_state_dict(weights["decoder"])
        
        # Use SD's VAE
        self.gcode_model.vae = self.sd_pipe.vae
        self.gcode_model.to(self.device)
        self.gcode_model.eval()
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("Models loaded!")
    
    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        max_gcode_tokens: int = 2048,
        temperature: float = 0.8,
        seed: int = None,
    ) -> tuple[str, Image.Image]:
        """Generate gcode from text prompt.
        
        Args:
            prompt: Text description
            num_inference_steps: SD diffusion steps
            guidance_scale: SD guidance scale
            max_gcode_tokens: Max gcode output length
            temperature: Gcode generation temperature
            seed: Random seed
            
        Returns:
            (gcode, image): Generated gcode string and intermediate image
        """
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Run SD to get latent (but also get the image for visualization)
        with torch.no_grad():
            # Get the latent before decoding to image
            latents = self.sd_pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="latent",
            ).images
            
            # Also decode to image for visualization
            image = self.sd_pipe.vae.decode(latents / self.sd_pipe.vae.config.scaling_factor).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = Image.fromarray((image * 255).astype("uint8"))
            
            # Scale latent properly for our model
            latents = latents * self.sd_pipe.vae.config.scaling_factor
        
        # Generate gcode from latent
        gcode_list = self.gcode_model.generate(
            latents=latents,
            tokenizer=self.tokenizer,
            max_new_tokens=max_gcode_tokens,
            temperature=temperature,
        )
        
        gcode = gcode_list[0]
        
        return gcode, image
    
    def generate_from_image(
        self,
        image: Image.Image,
        max_gcode_tokens: int = 2048,
        temperature: float = 0.8,
    ) -> str:
        """Generate gcode from an existing image (skip SD)."""
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Encode to latent
        with torch.no_grad():
            latent = self.gcode_model.encode_image(image_tensor)
        
        # Generate gcode
        gcode_list = self.gcode_model.generate(
            latents=latent,
            tokenizer=self.tokenizer,
            max_new_tokens=max_gcode_tokens,
            temperature=temperature,
        )
        
        return gcode_list[0]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate gcode from text")
    parser.add_argument("prompt", help="Text prompt")
    parser.add_argument("-m", "--model", required=True, help="Path to trained gcode model")
    parser.add_argument("--sd-model", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("-o", "--output", default="output.gcode", help="Output gcode file")
    parser.add_argument("--image-output", default="output.png", help="Output preview image")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    
    args = parser.parse_args()
    
    inferencer = DcodeInference(args.model, args.sd_model)
    
    gcode, image = inferencer.generate(
        args.prompt,
        num_inference_steps=args.steps,
        temperature=args.temperature,
        seed=args.seed,
    )
    
    # Save outputs
    Path(args.output).write_text(gcode)
    image.save(args.image_output)
    
    print(f"Gcode saved to {args.output}")
    print(f"Image saved to {args.image_output}")
    print(f"\nGcode preview:\n{gcode[:500]}...")


if __name__ == "__main__":
    main()

