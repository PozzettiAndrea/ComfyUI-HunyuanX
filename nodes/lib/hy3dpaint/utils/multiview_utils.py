# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import torch
import random
import numpy as np
from PIL import Image
from typing import List
import huggingface_hub
from omegaconf import OmegaConf
from diffusers import DiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, DDIMScheduler, UniPCMultistepScheduler
from ..hunyuanpaintpbr.pipeline import HunyuanPaintPipeline


class multiviewDiffusionNet:
    def __init__(self, config) -> None:
        self.device = config.device

        cfg_path = config.multiview_cfg_path
        custom_pipeline = config.custom_pipeline
        cfg = OmegaConf.load(cfg_path)
        self.cfg = cfg
        self.mode = self.cfg.model.params.stable_diffusion_config.custom_pipeline[2:]

        # Setup local cache directory inside custom node
        script_dir = os.path.dirname(os.path.abspath(__file__))
        node_root = os.path.dirname(os.path.dirname(script_dir))  # Go up 2 levels to ComfyUI-MeshCraft/
        cache_dir = os.path.join(node_root, "models", "hunyuan3d-multiview")
        os.makedirs(cache_dir, exist_ok=True)

        print(f"   📁 Model cache directory: {cache_dir}")

        # Try loading from cache first (no network calls)
        model_path = None
        try:
            print(f"   🔍 Checking for cached model...")
            model_path = huggingface_hub.snapshot_download(
                repo_id=config.multiview_pretrained_path,
                allow_patterns=["hunyuan3d-paintpbr-v2-1/**"],
                cache_dir=cache_dir,
                local_files_only=True,  # Only use cache, no network
            )
            print(f"   ✅ Using cached model (fast!)")
        except Exception:
            # Cache miss or incomplete - download from HuggingFace
            print(f"   📥 Downloading model from HuggingFace (first time)...")
            model_path = huggingface_hub.snapshot_download(
                repo_id=config.multiview_pretrained_path,
                allow_patterns=["hunyuan3d-paintpbr-v2-1/**"],
                cache_dir=cache_dir,
            )

        model_path = os.path.join(model_path, "hunyuan3d-paintpbr-v2-1")

        # Prepare pipeline loading args with attention mode
        pipeline_args = {
            "torch_dtype": torch.float16
        }

        # Map attention mode to diffusers attn_implementation
        attention_mode = getattr(config, 'attention_mode', 'sdpa')
        if attention_mode == "flash":
            pipeline_args["attn_implementation"] = "flash_attention_2"
            print(f"   🔥 Enabling Flash Attention 2 during pipeline load...")
        elif attention_mode == "sdpa":
            pipeline_args["attn_implementation"] = "sdpa"
            print(f"   ⚡ Using SDPA attention (PyTorch auto-select)...")

        pipeline = HunyuanPaintPipeline.from_pretrained(
            model_path,
            **pipeline_args
        )

        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
        pipeline.set_progress_bar_config(disable=False)
        pipeline.eval()
        setattr(pipeline, "view_size", cfg.model.params.get("view_size", 320))
        # pipeline.enable_model_cpu_offload()  # Disabled: Breaks model caching, keeps models on CPU after inference
        self.pipeline = pipeline.to(self.device)
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            from ..hunyuanpaintpbr.unet.modules import Dino_v2
            self.dino_v2 = Dino_v2(config.dino_ckpt_path).to(torch.float16)
            self.dino_v2 = self.dino_v2.to(self.device)

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    @torch.no_grad()
    def __call__(self, images, conditions, prompt=None, custom_view_size=None, resize_input=False, num_steps=10, guidance_scale=3.0, seed=0):
        pils = self.forward_one(
            images, conditions, prompt=prompt, custom_view_size=custom_view_size, resize_input=resize_input, num_steps=num_steps, guidance_scale=guidance_scale, seed=seed
        )
        return pils

    def forward_one(self, input_images, control_images, prompt=None, custom_view_size=None, resize_input=False, num_steps=10, guidance_scale=3.0, seed=0):
        self.seed_everything(seed)
        custom_view_size = custom_view_size if custom_view_size is not None else self.pipeline.view_size
        
        if not isinstance(input_images, List):
            input_images = [input_images]
            
        if not resize_input:
            input_images = [
                input_image.resize((self.pipeline.view_size, self.pipeline.view_size)) for input_image in input_images
            ]
        else:
            input_images = [input_image.resize((custom_view_size, custom_view_size)) for input_image in input_images]
            
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((custom_view_size, custom_view_size))
            if control_images[i].mode == "L":
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode="1")
        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        kwargs["width"] = custom_view_size
        kwargs["height"] = custom_view_size
        kwargs["num_in_batch"] = num_view
        kwargs["images_normal"] = normal_image
        kwargs["images_position"] = position_image

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            dino_hidden_states = self.dino_v2(input_images[0])
            kwargs["dino_hidden_states"] = dino_hidden_states

        sync_condition = None

        infer_steps_dict = {
            "EulerAncestralDiscreteScheduler": 10,
            "UniPCMultistepScheduler": 10,
            "DDIMScheduler": 10,
            "ShiftSNRScheduler": 10,
        }

        mvd_image = self.pipeline(
            input_images[0:1],
            num_inference_steps=num_steps,
            prompt=prompt,
            sync_condition=sync_condition,
            guidance_scale=guidance_scale,
            **kwargs,
        ).images

        if "pbr" in self.mode:
            mvd_image = {"albedo": mvd_image[:num_view], "mr": mvd_image[num_view:]}
            # mvd_image = {'albedo':mvd_image[:num_view]}
        else:
            mvd_image = {"hdr": mvd_image}

        return mvd_image
