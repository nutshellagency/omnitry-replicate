import os
import sys
import torch
import random
import math
import numpy as np
import torchvision.transforms as T
import peft
from peft import LoraConfig
from safetensors import safe_open
from omegaconf import OmegaConf
from cog import BasePredictor, Input, Path
from PIL import Image

sys.path.append("/omnitry")
from omnitry.models.transformer_flux import FluxTransformer2DModel
from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline

# Match gradio demo map
OBJECT_MAP = {
    "upper_body": "top clothes",
    "lower_body": "bottom clothes",
    "dresses": "dresses",
    "accessories": "accessories",
    "shoes": "shoes",
    "scarves": "scarves",
    "bags": "bags",
    "jewelry": "jewelry"
}

def create_hacked_forward(module):
    def lora_forward(self, active_adapter, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        if active_adapter is not None:
            torch_result_dtype = result.dtype
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)
            result = result + lora_B(lora_A(dropout(x))) * scaling
        return result

    def hacked_lora_forward(self, x, *args, **kwargs):
        return torch.cat((
            lora_forward(self, 'vtryon_lora', x[:1], *args, **kwargs),
            lora_forward(self, 'garment_lora', x[1:], *args, **kwargs),
        ), dim=0)

    return hacked_lora_forward.__get__(module, type(module))

class Predictor(BasePredictor):
    def setup(self):
        print("Downloading OmniTry models from HuggingFace directly to the Replicate node...")
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
        from huggingface_hub import snapshot_download
        import time
        
        def safe_download(repo_id, local_dir, max_retries=15):
            for i in range(max_retries):
                try:
                    snapshot_download(repo_id=repo_id, local_dir=local_dir, resume_download=True)
                    return
                except Exception as e:
                    print(f"Download {repo_id} failed on attempt {i+1} with error: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
            raise RuntimeError(f"Failed to download {repo_id} after {max_retries} attempts. HuggingFace connection keeps dropping.")
            
        self.device = torch.device('cuda:0')
        self.weight_dtype = torch.bfloat16
        
        weights_root = "/tmp/omnitry_weights/flux"
        os.makedirs(weights_root, exist_ok=True)
        print("Downloading FLUX base model...")
        safe_download(repo_id='camenduru/FLUX.1-Fill-dev-ungated', local_dir=weights_root)
        
        lora_dir = "/tmp/omnitry_weights/checkpoints"
        os.makedirs(lora_dir, exist_ok=True)
        print("Downloading LoRA weights...")
        safe_download(repo_id='Kunbyte/OmniTry', local_dir=lora_dir)
        
        print("Loading OmniTry models into VRAM...")
        
        # Load logic from gradio_demo.py
        print("Initializing FluxTransformer2DModel...")
        self.transformer = FluxTransformer2DModel.from_pretrained(f"{weights_root}", subfolder="transformer").requires_grad_(False).to(dtype=self.weight_dtype)
        
        print("Initializing FluxFillPipeline...")
        self.pipeline = FluxFillPipeline.from_pretrained(f"{weights_root}", transformer=self.transformer.eval(), torch_dtype=self.weight_dtype)
        
        # VRAM saving configurations
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.vae.enable_tiling()
        
        # Insert LoRA
        print("Loading LoRA Configurations...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            init_lora_weights="gaussian",
            target_modules=[
                'x_embedder',
                'attn.to_k', 'attn.to_q', 'attn.to_v', 'attn.to_out.0', 
                'attn.add_k_proj', 'attn.add_q_proj', 'attn.add_v_proj', 'attn.to_add_out', 
                'ff.net.0.proj', 'ff.net.2', 'ff_context.net.0.proj', 'ff_context.net.2', 
                'norm1_context.linear', 'norm1.linear', 'norm.linear', 'proj_mlp', 'proj_out'
            ]
        )
        self.transformer.add_adapter(lora_config, adapter_name='vtryon_lora')
        self.transformer.add_adapter(lora_config, adapter_name='garment_lora')

        print("Loading Safetensors...")
        lora_path = "/tmp/omnitry_weights/checkpoints/omnitry_v1_unified.safetensors"
        with safe_open(lora_path, framework="pt") as f:
            lora_weights = {k: f.get_tensor(k) for k in f.keys()}
            self.transformer.load_state_dict(lora_weights, strict=False)

        print("Hacking LoRA forward...")
        for n, m in self.transformer.named_modules():
            if isinstance(m, peft.tuners.lora.layer.Linear):
                m.forward = create_hacked_forward(m)
        
        print("Model setup complete.")

    def seed_everything(self, seed=0):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def predict(
        self,
        human_img: Path = Input(description="Input image of the person"),
        garm_img: Path = Input(description="Input image of the garment/clothing"),
        garment_category: str = Input(
            description="Category of the garment", 
            default="upper_body",
            choices=["upper_body", "lower_body", "dresses", "accessories", "shoes"]
        ),
        num_inference_steps: int = Input(description="Number of denoising steps", default=30),
        guidance_scale: float = Input(description="Guidance scale", default=7.5),
        seed: int = Input(description="Random seed (-1 for random)", default=-1)
    ) -> Path:
        """Run OmniTry virtual try-on prediction"""
        print(f"Running OmniTry prediction for category: {garment_category}")
        
        person_image = Image.open(str(human_img)).convert("RGB")
        object_image = Image.open(str(garm_img)).convert("RGB")
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        self.seed_everything(seed)

        # Resize person image bounds limit
        max_area = 1024 * 1024
        oW = person_image.width
        oH = person_image.height
        ratio = math.sqrt(max_area / (oW * oH))
        ratio = min(1, ratio)
        tW, tH = int(oW * ratio) // 16 * 16, int(oH * ratio) // 16 * 16
        transform_person = T.Compose([
            T.Resize((tH, tW)),
            T.ToTensor(),
        ])
        person_tensor = transform_person(person_image)

        # Resize and pad garment
        ratio_garm = min(tW / object_image.width, tH / object_image.height)
        transform_garm = T.Compose([
            T.Resize((int(object_image.height * ratio_garm), int(object_image.width * ratio_garm))),
            T.ToTensor(),
        ])
        object_image_padded = torch.ones_like(person_tensor)
        object_tensor = transform_garm(object_image)
        new_h, new_w = object_tensor.shape[1], object_tensor.shape[2]
        min_x = (tW - new_w) // 2
        min_y = (tH - new_h) // 2
        object_image_padded[:, min_y: min_y + new_h, min_x: min_x + new_w] = object_tensor

        # Prepare prompts & conditions
        mapped_class = OBJECT_MAP.get(garment_category, "top clothes")
        prompts = [mapped_class] * 2
        img_cond = torch.stack([person_tensor, object_image_padded]).to(dtype=self.weight_dtype, device=self.device) 
        mask = torch.zeros_like(img_cond).to(img_cond)

        print(f"Executing pipeline with shape {tW}x{tH}...")
        with torch.no_grad():
            img = self.pipeline(
                prompt=prompts,
                height=tH,
                width=tW,    
                img_cond=img_cond,
                mask=mask,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(self.device).manual_seed(seed),
            ).images[0]

        output_path = "/tmp/output.png"
        img.save(output_path)
        
        print(f"Generation complete. Saved to {output_path}")
        return Path(output_path)
