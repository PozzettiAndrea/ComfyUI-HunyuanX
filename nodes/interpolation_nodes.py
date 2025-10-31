"""
ComfyUI-MeshCraft Interpolation Nodes

Image and 3D mesh interpolation nodes for smooth transitions between two inputs.
Supports VAE latent interpolation (2D) and Hunyuan3D latent interpolation (3D).

Implements SLERP (Spherical Linear Interpolation) - the best practice for latent space
interpolation, as recommended by the stable diffusion community.

References:
- SLERP implementation: https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355
- ComfyUI built-in LatentInterpolate node
"""

import torch
import numpy as np
from typing import Tuple, List
import folder_paths
import comfy.model_management as mm
import comfy.utils

# =============================================================================
# Utility Functions - SLERP and LERP
# =============================================================================

def slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """
    Spherical Linear Interpolation (SLERP) between two tensors.

    SLERP is the preferred method for latent space interpolation because it:
    - Preserves the magnitude of vectors
    - Provides smoother transitions
    - Avoids statistical issues that cause gray images with LERP

    Based on: https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355

    Args:
        val: Interpolation factor (0.0 to 1.0)
        low: Starting tensor
        high: Ending tensor

    Returns:
        Interpolated tensor
    """
    low_norm = low / torch.norm(low, dim=-1, keepdim=True)
    high_norm = high / torch.norm(high, dim=-1, keepdim=True)

    # Calculate angle between vectors
    omega = torch.acos((low_norm * high_norm).sum(-1))
    so = torch.sin(omega)

    # Handle near-zero angles (vectors are nearly parallel)
    # Fall back to LERP to avoid division by zero
    if torch.abs(so) < 1e-6:
        return (1.0 - val) * low + val * high

    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(-1) * low + \
          (torch.sin(val * omega) / so).unsqueeze(-1) * high

    return res


def lerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """
    Linear Interpolation (LERP) between two tensors.

    Note: LERP can cause statistical issues in latent spaces.
    Use SLERP for better results in most cases.

    Args:
        val: Interpolation factor (0.0 to 1.0)
        low: Starting tensor
        high: Ending tensor

    Returns:
        Interpolated tensor
    """
    return (1.0 - val) * low + val * high


def batch_slerp(val: float, low_batch: torch.Tensor, high_batch: torch.Tensor) -> torch.Tensor:
    """
    Apply SLERP to batches of tensors.

    Args:
        val: Interpolation factor (0.0 to 1.0)
        low_batch: Starting batch tensor (B, ...)
        high_batch: Ending batch tensor (B, ...)

    Returns:
        Interpolated batch tensor
    """
    # Flatten batch dimensions for SLERP
    batch_size = low_batch.shape[0]
    low_flat = low_batch.reshape(batch_size, -1)
    high_flat = high_batch.reshape(batch_size, -1)

    # Apply SLERP
    result_flat = slerp(val, low_flat, high_flat)

    # Restore original shape
    return result_flat.reshape_as(low_batch)


# =============================================================================
# Node 1: VAE Latent Interpolation (2D Images)
# =============================================================================

class VAELatentInterpolation:
    """
    Interpolate between two images in VAE latent space.

    Encodes both images to VAE latents, interpolates using SLERP or LERP,
    then decodes back to a single image at the specified factor.

    Best for: 2D image transitions, visual morphing
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "vae": ("VAE",),
                "factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "0.0 = image1, 1.0 = image2"
                }),
                "interpolation_method": (["slerp", "lerp"], {
                    "default": "slerp"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "interpolate"
    CATEGORY = "MeshCraft/Interpolation"

    def interpolate(self, image1, image2, vae, factor, interpolation_method):
        """
        Perform VAE latent space interpolation.

        Args:
            image1: Starting image tensor (B, H, W, C)
            image2: Ending image tensor (B, H, W, C)
            vae: VAE model for encoding/decoding
            factor: Interpolation factor (0.0 to 1.0)
            interpolation_method: "slerp" or "lerp"

        Returns:
            Tuple containing single interpolated image
        """
        print(f"\n🔄 VAE Latent Interpolation:")
        print(f"   Method: {interpolation_method.upper()}")
        print(f"   Factor: {factor:.3f}")

        # Encode images to latent space
        latent1 = vae.encode(image1[:, :, :, :3])
        latent2 = vae.encode(image2[:, :, :, :3])

        # Get latent samples
        samples1 = latent1["samples"]
        samples2 = latent2["samples"]

        print(f"   Latent shape: {samples1.shape}")

        # Ensure both latents have the same shape
        if samples1.shape != samples2.shape:
            raise ValueError(f"Latent shapes don't match: {samples1.shape} vs {samples2.shape}")

        # Interpolate latents
        if interpolation_method == "slerp":
            interp_latent = batch_slerp(factor, samples1, samples2)
        else:
            interp_latent = lerp(factor, samples1, samples2)

        print(f"   Interpolated latent shape: {interp_latent.shape}")

        # Decode latent back to image
        decoded_image = vae.decode(interp_latent)

        print(f"   Output image shape: {decoded_image.shape}")
        print("=== End VAE Interpolation ===\n")

        return (decoded_image,)


# =============================================================================
# Node 2: Hunyuan3D Latent Interpolation (3D Meshes)
# =============================================================================

class Hunyuan3DLatentInterpolation:
    """
    Interpolate between two Hunyuan3D latents.

    Takes two HY3DLATENT outputs and returns a single interpolated latent
    based on the factor (0.0 = latent1, 1.0 = latent2).

    Best for: 3D mesh transitions, geometric morphing
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent1": ("HY3DLATENT",),
                "latent2": ("HY3DLATENT",),
                "factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "0.0 = latent1, 1.0 = latent2"
                }),
                "interpolation_method": (["slerp", "lerp"], {
                    "default": "slerp"
                }),
            },
        }

    RETURN_TYPES = ("HY3DLATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "interpolate"
    CATEGORY = "MeshCraft/Interpolation"

    def interpolate(self, latent1, latent2, factor, interpolation_method):
        """
        Perform Hunyuan3D latent space interpolation.

        Args:
            latent1: Starting Hunyuan3D latent tensor
            latent2: Ending Hunyuan3D latent tensor
            factor: Interpolation factor (0.0 to 1.0)
            interpolation_method: "slerp" or "lerp"

        Returns:
            Tuple containing single interpolated Hunyuan3D latent
        """
        # Ensure latents are torch tensors
        if not isinstance(latent1, torch.Tensor):
            latent1 = torch.tensor(latent1)
        if not isinstance(latent2, torch.Tensor):
            latent2 = torch.tensor(latent2)

        # Ensure both latents have the same shape
        if latent1.shape != latent2.shape:
            raise ValueError(f"Latent shapes don't match: {latent1.shape} vs {latent2.shape}")

        print(f"\n🔄 Hunyuan3D Latent Interpolation:")
        print(f"   Method: {interpolation_method.upper()}")
        print(f"   Factor: {factor:.3f}")
        print(f"   Latent shape: {latent1.shape}")

        # Interpolate
        if interpolation_method == "slerp":
            if len(latent1.shape) > 1:
                result = batch_slerp(factor, latent1, latent2)
            else:
                result = slerp(factor, latent1, latent2)
        else:
            result = lerp(factor, latent1, latent2)

        print(f"   Output shape: {result.shape}")
        print("=== End Hunyuan3D Interpolation ===\n")

        return (result,)


# =============================================================================
# Node 3: DINO Embedding Interpolation
# =============================================================================

class DINOEmbeddingInterpolation:
    """
    Interpolate between two DINO v2 embeddings.

    Takes two DINO_EMBEDDING outputs from EncodeDINO nodes and returns
    a single interpolated embedding based on the factor (0.0 = embedding1, 1.0 = embedding2).

    Best for: Exploring semantic transitions, guided 3D generation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embedding1": ("DINO_EMBEDDING",),
                "embedding2": ("DINO_EMBEDDING",),
                "factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "0.0 = embedding1, 1.0 = embedding2"
                }),
                "interpolation_method": (["slerp", "lerp"], {
                    "default": "slerp"
                }),
            },
        }

    RETURN_TYPES = ("DINO_EMBEDDING",)
    RETURN_NAMES = ("embedding",)
    FUNCTION = "interpolate"
    CATEGORY = "MeshCraft/Interpolation"

    def interpolate(self, embedding1, embedding2, factor, interpolation_method):
        """
        Perform DINO embedding interpolation.

        Args:
            embedding1: Starting DINO embedding dict
            embedding2: Ending DINO embedding dict
            factor: Interpolation factor (0.0 to 1.0)
            interpolation_method: "slerp" or "lerp"

        Returns:
            Tuple containing single interpolated DINO embedding
        """
        print(f"\n🔄 DINO Embedding Interpolation:")
        print(f"   Method: {interpolation_method.upper()}")
        print(f"   Factor: {factor:.3f}")
        print(f"   Embedding 1 type: {type(embedding1)}")
        print(f"   Embedding 2 type: {type(embedding2)}")

        # DINO embeddings are dictionaries with tensors
        if isinstance(embedding1, dict) and isinstance(embedding2, dict):
            # Find matching keys
            keys1 = set(embedding1.keys())
            keys2 = set(embedding2.keys())
            common_keys = keys1 & keys2

            print(f"   Common keys: {list(common_keys)}")

            if not common_keys:
                raise ValueError("DINO embeddings have no common keys!")

            # Interpolate each common key
            interp_dict = {}

            for key in common_keys:
                val1 = embedding1[key]
                val2 = embedding2[key]

                # Skip non-tensor values
                if not isinstance(val1, torch.Tensor) or not isinstance(val2, torch.Tensor):
                    interp_dict[key] = val1
                    continue

                # Check shape compatibility
                if val1.shape != val2.shape:
                    print(f"   Warning: Shape mismatch for key '{key}': {val1.shape} vs {val2.shape}")
                    interp_dict[key] = val1
                    continue

                # Interpolate tensor
                if interpolation_method == "slerp":
                    if len(val1.shape) > 1 and val1.shape[0] == 1:
                        interp_tensor = batch_slerp(factor, val1, val2)
                    else:
                        interp_tensor = slerp(factor, val1, val2)
                else:
                    interp_tensor = lerp(factor, val1, val2)

                interp_dict[key] = interp_tensor
                print(f"   Interpolated '{key}': {val1.shape}")

            print("=== End DINO Interpolation ===\n")
            return (interp_dict,)

        else:
            # Fallback: treat as raw tensors
            print("   Note: Embeddings are not dicts, treating as raw tensors")

            if not isinstance(embedding1, torch.Tensor):
                embedding1 = torch.tensor(embedding1)
            if not isinstance(embedding2, torch.Tensor):
                embedding2 = torch.tensor(embedding2)

            if embedding1.shape != embedding2.shape:
                raise ValueError(f"Embedding shapes don't match: {embedding1.shape} vs {embedding2.shape}")

            # Interpolate
            if interpolation_method == "slerp":
                if len(embedding1.shape) > 1:
                    result = batch_slerp(factor, embedding1, embedding2)
                else:
                    result = slerp(factor, embedding1, embedding2)
            else:
                result = lerp(factor, embedding1, embedding2)

            return (result,)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "MeshCraft_VAELatentInterpolation": VAELatentInterpolation,
    "MeshCraft_Hunyuan3DLatentInterpolation": Hunyuan3DLatentInterpolation,
    "MeshCraft_DINOEmbeddingInterpolation": DINOEmbeddingInterpolation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshCraft_VAELatentInterpolation": "VAE Latent Interpolation",
    "MeshCraft_Hunyuan3DLatentInterpolation": "Hunyuan3D Latent Interpolation",
    "MeshCraft_DINOEmbeddingInterpolation": "DINO Embedding Interpolation",
}
