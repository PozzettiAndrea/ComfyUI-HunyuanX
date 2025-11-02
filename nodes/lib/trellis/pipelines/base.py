from typing import *
import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/pipeline.json")

        if is_local:
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, "pipeline.json")

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            # Detect if path is already a full repo reference (e.g., "JeffreyXiang/TRELLIS-image-large/ckpts/...")
            # or a relative path within the current repo (e.g., "ckpts/...")
            try:
                # Count path components: if it has 3+ parts and starts with a repo name, it's a full reference
                path_parts = v.split('/')
                if len(path_parts) >= 3 and '.' not in path_parts[0]:
                    # Full repo path: replace JeffreyXiang with microsoft (Microsoft forked the repo)
                    model_path = v.replace("JeffreyXiang/", "microsoft/")
                    print(f"[Trellis] Loading model '{k}' from cross-repo: {model_path}")
                else:
                    # Relative path: prepend current repo (e.g., "ckpts/model")
                    model_path = f"{path}/{v}"
                    print(f"[Trellis] Loading model '{k}' from: {model_path}")

                _models[k] = models.from_pretrained(model_path)
                print(f"[Trellis] ✓ Successfully loaded model '{k}'")
            except Exception as e:
                print(f"[Trellis] ❌ Failed to load '{k}' from '{model_path}'")
                print(f"[Trellis]    Error: {type(e).__name__}: {e}")
                print(f"[Trellis]    This is a fatal error - cannot continue without all models")
                raise  # Re-raise the exception instead of trying a broken fallback

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
