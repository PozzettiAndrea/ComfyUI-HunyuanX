# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
TRELLIS Utilities

Merged from ComfyUI_TRELLIS utils.py and app.py
Contains utility functions for TRELLIS 3D generation pipeline.
"""

import os
import uuid
import torch
import random
from PIL import Image
import numpy as np
import cv2
import sys
import trimesh
from typing import *
import imageio
from easydict import EasyDict as edict

from comfy.utils import common_upscale, ProgressBar
import folder_paths
from ..lib.trellis.utils import render_utils, postprocessing_utils
from ..lib.trellis.representations import Gaussian, MeshExtractResult

MAX_SEED = np.iinfo(np.int32).max
current_path = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# TRELLIS Pipeline Functions (from app.py)
# ============================================================================

def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    """Pack Gaussian and Mesh state for serialization"""
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }


def unpack_state(state: dict) -> Tuple[Gaussian, edict]:
    """Unpack Gaussian and Mesh state from serialization"""
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')

    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )

    return gs, mesh


def get_seed(randomize_seed: bool, seed: int) -> int:
    """Get the random seed"""
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def image_to_3d(pipeline, image, preprocess_image: bool, covert2video: bool, trial_id: str, seed: int,
                ss_guidance_strength: float, ss_sampling_steps: int, slat_guidance_strength: float,
                slat_sampling_steps: int, mesh_simplify, texture_size, mode, is_multiimage, gaussians2ply,
                multiimage_algo):
    """
    Convert an image to a 3D model.

    Args:
        pipeline: TRELLIS pipeline
        image: Input image(s)
        preprocess_image: Whether to preprocess the image
        covert2video: Whether to render videos
        trial_id: UUID of the trial
        seed: Random seed
        ss_guidance_strength: Guidance strength for sparse structure generation
        ss_sampling_steps: Number of sampling steps for sparse structure generation
        slat_guidance_strength: Guidance strength for structured latent generation
        slat_sampling_steps: Number of sampling steps for structured latent generation
        mesh_simplify: Mesh simplification ratio
        texture_size: Texture size
        mode: Processing mode
        is_multiimage: Whether using multi-image input
        gaussians2ply: Whether to save PLY file
        multiimage_algo: Multi-image fusion algorithm

    Returns:
        str: Path to generated GLB file
    """
    if not is_multiimage:
        outputs = pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=preprocess_image,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
    else:
        outputs = pipeline.run_multi_image(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=preprocess_image,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
            mode=multiimage_algo,
        )

    return prepare_output(outputs, covert2video, trial_id, is_multiimage, gaussians2ply, mesh_simplify, texture_size,
                          mode)


def prepare_output(outputs, covert2video, trial_id, is_multiimage, gaussians2ply, mesh_simplify, texture_size, mode):
    """Prepare and export TRELLIS output to GLB file"""
    print(f"[Debug prepare_output] Starting output preparation")

    if covert2video:
        print(f"[Debug prepare_output] Rendering videos...")
        video_path = f"{trial_id}.mp4"
        if is_multiimage:
            video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
            video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
            video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in
                     zip(video_gs, video_mesh)]
            imageio.mimsave(video_path, video, fps=30)
        else:
            video = render_utils.render_video(outputs['gaussian'][0])['color']
            imageio.mimsave(f"{trial_id}_gs.mp4", video, fps=30)
            video = render_utils.render_video(outputs['radiance_field'][0])['color']
            imageio.mimsave(f"{trial_id}_rf.mp4", video, fps=30)
            video = render_utils.render_video(outputs['mesh'][0])['normal']
            imageio.mimsave(f"{trial_id}_mesh.mp4", video, fps=30)
        print(f"[Debug prepare_output] Videos rendered")

    print(f"[Debug prepare_output] Packing state...")
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])

    print(f"[Debug prepare_output] Clearing CUDA cache...")
    torch.cuda.empty_cache()

    print(f"[Debug prepare_output] Unpacking state...")
    gs, mesh = unpack_state(state)

    if gaussians2ply:
        print(f"[Debug prepare_output] Saving PLY...")
        gs.save_ply(f"{folder_paths.get_output_directory()}/{trial_id}.ply")

    print(f"[Debug prepare_output] Converting to GLB (simplify={mesh_simplify}, texture_size={texture_size}, mode={mode})...")
    print(f"[Debug prepare_output] Mesh vertices: {mesh.vertices.shape}, faces: {mesh.faces.shape}")

    try:
        glb = postprocessing_utils.to_glb(
            gs,
            mesh,
            simplify=mesh_simplify,
            texture_size=texture_size,
            mode=mode,
            uv_map=f"{folder_paths.get_output_directory()}/{trial_id}_uv_map.png",
        )
        print(f"[Debug prepare_output] ✓ GLB conversion completed")
    except Exception as e:
        print(f"[Debug prepare_output] ❌ GLB conversion failed: {type(e).__name__}: {e}")
        raise

    prefix = ''.join(random.choice("0123456789") for _ in range(5))
    glb_path = f"{folder_paths.get_output_directory()}/{trial_id}_{prefix}.glb"

    print(f"[Debug prepare_output] Exporting GLB to: {glb_path}")
    glb.export(glb_path)
    print(f"[Debug prepare_output] ✓ GLB exported successfully")
    print(f"glb save in {glb_path} ")
    return glb_path


# ============================================================================
# General Utility Functions (from utils.py)
# ============================================================================

def glb2obj_(glb_path, obj_path):
    """Convert GLB to OBJ format"""
    print('Converting glb to obj')
    mesh = trimesh.load(glb_path)

    if isinstance(mesh, trimesh.Scene):
        vertices = 0
        for g in mesh.geometry.values():
            vertices += g.vertices.shape[0]
    elif isinstance(mesh, trimesh.Trimesh):
        vertices = mesh.vertices.shape[0]
    else:
        raise ValueError('It is not mesh or scene')

    if vertices > 300000:
        raise ValueError('Too many vertices')
    if not os.path.exists(os.path.dirname(obj_path)):
        os.makedirs(os.path.dirname(obj_path))
    mesh.export(obj_path)
    print('Convert Done')


def obj2fbx_(obj_path, fbx_path):
    """Convert OBJ to FBX format"""
    import bpy
    if "Cube" in bpy.data.meshes:
        mesh = bpy.data.meshes["Cube"]
        print("removing mesh", mesh)
        bpy.data.meshes.remove(mesh)

    bpy.ops.import_scene.obj(filepath=obj_path)
    bpy.ops.export_scene.fbx(filepath=fbx_path)
    print('Convert Done')


def preprocess_image_(image: Image.Image, pipe, TMP_DIR):
    """Preprocess the input image"""
    trial_id = str(uuid.uuid4())
    processed_image = pipe.preprocess_image(image)
    processed_image.save(f"{TMP_DIR}/{trial_id}.png")
    return trial_id, processed_image


def find_directories(base_path):
    """Find all directories under base_path"""
    directories = []
    for root, dirs, files in os.walk(base_path):
        for name in dirs:
            directories.append(name)
    return directories


def pil2narry(img):
    """Convert PIL image to numpy array tensor"""
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img


def narry_list(list_in):
    """Convert list of PIL images to list of numpy array tensors"""
    for i in range(len(list_in)):
        value = list_in[i]
        modified_value = pil2narry(value)
        list_in[i] = modified_value
    return list_in


def get_video_img(tensor):
    """Extract images from video tensor"""
    if tensor == None:
        return None
    outputs = []
    for x in tensor:
        x = tensor_to_pil(x)
        outputs.append(x)
    yield outputs


def instance_path(path, repo):
    """Get instance path for model"""
    if repo == "":
        if path == "none":
            repo = "none"
        else:
            model_path = get_local_path(folder_paths.base_path, path)
            repo = get_instance_path(model_path)
    return repo


def gen_img_form_video(tensor):
    """Generate images from video tensor"""
    pil = []
    for x in tensor:
        pil[x] = tensor_to_pil(x)
    yield pil


def phi_list(list_in):
    """Process list (identity function)"""
    for i in range(len(list_in)):
        value = list_in[i]
        list_in[i] = value
    return list_in


def tensor_to_pil(tensor):
    """Convert tensor to PIL image"""
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image


def nomarl_upscale(img_tensor, width, height):
    """Upscale tensor and convert to PIL"""
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor_to_pil(samples)
    return img_pil


def tensor_upscale(img_tensor, width, height):
    """Upscale tensor"""
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    return samples


def get_local_path(comfy_file_path, model_path):
    """Get local path for model"""
    path = os.path.join(comfy_file_path, "models", "diffusers", model_path)
    model_path = os.path.normpath(path)
    if sys.platform == 'win32':
        model_path = model_path.replace('\\', "/")
    return model_path


def get_instance_path(path):
    """Normalize instance path"""
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path


def tensor2cv(tensor_image):
    """Convert tensor to OpenCV image"""
    if len(tensor_image.shape) == 4:  # b hwc to hwc
        tensor_image = tensor_image.squeeze(0)
    if tensor_image.is_cuda:
        tensor_image = tensor_image.cpu()
    tensor_image = tensor_image.numpy()
    # 反归一化
    maxValue = tensor_image.max()
    tensor_image = tensor_image * 255 / maxValue
    img_cv2 = np.uint8(tensor_image)  # 32 to uint8
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
    return img_cv2


def cvargb2tensor(img):
    """Convert OpenCV RGB image to tensor"""
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)


def cv2tensor(img):
    """Convert OpenCV BGR image to tensor"""
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img)
    return img.float().div(255).unsqueeze(0)


def images_generator(img_list: list, ):
    """Generate normalized images from list"""
    # get img size
    sizes = {}
    for image_ in img_list:
        if isinstance(image_, Image.Image):
            count = sizes.get(image_.size, 0)
            sizes[image_.size] = count + 1
        elif isinstance(image_, np.ndarray):
            count = sizes.get(image_.shape[:2][::-1], 0)
            sizes[image_.shape[:2][::-1]] = count + 1
        else:
            raise "unsupport image list,must be pil or cv2!!!"
    size = max(sizes.items(), key=lambda x: x[1])[0]
    yield size[0], size[1]

    # any to tensor
    def load_image(img_in):
        if isinstance(img_in, Image.Image):
            img_in = img_in.convert("RGB")
            i = np.array(img_in, dtype=np.float32)
            i = torch.from_numpy(i).div_(255)
            if i.shape[0] != size[1] or i.shape[1] != size[0]:
                i = torch.from_numpy(i).movedim(-1, 0).unsqueeze(0)
                i = common_upscale(i, size[0], size[1], "lanczos", "center")
                i = i.squeeze(0).movedim(0, -1).numpy()
            return i
        elif isinstance(img_in, np.ndarray):
            i = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB).astype(np.float32)
            i = torch.from_numpy(i).div_(255)
            return i
        else:
            raise "unsupport image list,must be pil,cv2 or tensor!!!"

    total_images = len(img_list)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(load_image, img_list)
    try:
        prev_image = next(images)
        while True:
            next_image = next(images)
            yield prev_image
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image = next_image
    except StopIteration:
        pass
    if prev_image is not None:
        yield prev_image


def load_images(img_list: list, ):
    """Load list of images into tensor"""
    gen = images_generator(img_list)
    (width, height) = next(gen)
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded .")
    return images


def tensor2pil(tensor):
    """Convert tensor to PIL image"""
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image


def tensor2imglist(image):
    """Convert tensor to list of PIL images"""
    B, _, _, _ = image.size()
    if B == 1:
        list_out = [tensor2pil_preprocess(image)]
    else:
        image_list = torch.chunk(image, chunks=B)
        list_out = [tensor2pil_preprocess(i) for i in image_list]
    return list_out, B


def tensor2pil_preprocess(image):
    """Convert tensor to preprocessed PIL image"""
    cv_image = tensor2cv(image)
    cv_image = center_resize_pad(cv_image, 512, 512)
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    iamge_pil = Image.fromarray(img)
    return iamge_pil


def cf_tensor2cv(tensor, width, height):
    """Convert ComfyUI tensor to OpenCV image"""
    d1, _, _, _ = tensor.size()
    if d1 > 1:
        tensor_list = list(torch.chunk(tensor, chunks=d1))
        tensor = [tensor_list][0]
    cr_tensor = tensor_upscale(tensor, width, height)
    cv_img = tensor2cv(cr_tensor)
    return cv_img


def pre_img(tensor, max):
    """Preprocess image tensor"""
    cv_image = tensor2cv(tensor)  # 转CV
    h, w = cv_image.shape[:2]
    cv_image = center_resize_pad(cv_image, h, h)  # 以高度中心裁切或填充
    cv_image = cv2.resize(cv_image, (max, max))  # 缩放到统一高度
    return cv2tensor(cv_image)


def center_resize_pad(img, new_width, new_height):
    """Center crop or pad image to new dimensions"""
    h, w = img.shape[:2]
    if w == h:
        if w == new_width:
            return img
        else:
            return cv2.resize(img, (new_width, new_height))
    else:  # 蒙版也有可能不是正方形
        if h > w:  # 竖直图左右填充
            s = max(h, w)
            f = np.zeros((s, s, 3), np.uint8)
            ax, ay = (s - img.shape[1]) // 2, (s - img.shape[0]) // 2
            f[ay:img.shape[0] + ay, ax:ax + img.shape[1]] = img
        else:
            f = center_crop(img, h, h)
        return cv2.resize(f, (new_width, new_height))


def center_crop(image, crop_width, crop_height):
    """Center crop image"""
    # 获取图像的中心坐标
    height, width = image.shape[:2]
    x = width // 2 - crop_width // 2
    y = height // 2 - crop_height // 2

    x = max(0, x)
    y = max(0, y)

    # 裁剪图像
    crop_img = image[y:y + crop_height, x:x + crop_width]
    return crop_img
