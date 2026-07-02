"""
Anima Tiled Spatial Upscaler - Helper Utilities.
Contains cropping, masking, noise slicing, and RoPE positional embedder patching functions.
"""

import math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.conds
import comfy.model_management
from nodes import common_ksampler


# ---------------------------------------------------------------------------
# Tensor / PIL conversions
# ---------------------------------------------------------------------------

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a PyTorch tensor scaled to [0, 1]."""
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_pil(tensor: torch.Tensor, index: int = 0) -> Image.Image:
    """Convert a PyTorch tensor (at index) back to a PIL Image."""
    arr = tensor[index].cpu().numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Conditioning Cropping Helpers
# ---------------------------------------------------------------------------

def create_latent_blend_mask(tile_h: int, tile_w: int, lcx1: int, lcy1: int,
                             lcx2: int, lcy2: int, blend_size: int,
                             device: torch.device) -> torch.Tensor:
    """
    Create a 2D blend mask in latent space to smoothly interpolate overlapping seams.
    """
    if blend_size <= 0:
        mask = torch.zeros((tile_h, tile_w), dtype=torch.float32, device=device)
        mask[lcy1:lcy2, lcx1:lcx2] = 1.0
        return mask

    y_indices = torch.arange(tile_h, dtype=torch.float32, device=device).view(-1, 1)
    x_indices = torch.arange(tile_w, dtype=torch.float32, device=device).view(1, -1)

    # Distance of each pixel to the boundary of the core region
    dist_left = ((x_indices - (lcx1 - blend_size)) / blend_size).clamp(0.0, 1.0)
    dist_right = (((lcx2 + blend_size) - x_indices) / blend_size).clamp(0.0, 1.0)
    dist_top = ((y_indices - (lcy1 - blend_size)) / blend_size).clamp(0.0, 1.0)
    dist_bottom = (((lcy2 + blend_size) - y_indices) / blend_size).clamp(0.0, 1.0)

    t_x = dist_left * dist_right
    t_y = dist_top * dist_bottom

    smooth_x = 0.5 - 0.5 * torch.cos(t_x * math.pi)
    smooth_y = 0.5 - 0.5 * torch.cos(t_y * math.pi)

    return smooth_x * smooth_y


def crop_tensor(tensor: torch.Tensor, region: tuple) -> torch.Tensor:
    """Slice a PyTorch tensor (B, H, W, C) dynamically using region coordinates (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = region
    return tensor[:, y1:y2, x1:x2, :]


def resize_tensor(tensor: torch.Tensor, size: tuple, mode: str = "nearest-exact") -> torch.Tensor:
    """Resize a PyTorch tensor to the specified size."""
    return F.interpolate(tensor, size=size, mode=mode)


def resize_region(region: tuple, init_size: tuple, resize_size: tuple) -> tuple:
    """Map coordinates of a region from an initial size to a resized size."""
    x1, y1, x2, y2 = region
    init_width, init_height = init_size
    resize_width, resize_height = resize_size
    x1 = math.floor(x1 * resize_width / init_width)
    x2 = math.ceil(x2 * resize_width / init_width)
    y1 = math.floor(y1 * resize_height / init_height)
    y2 = math.ceil(y2 * resize_height / init_height)
    return (x1, y1, x2, y2)


def pad_image2(image: Image.Image, left_pad: int, right_pad: int,
               top_pad: int, bottom_pad: int, fill: bool = False,
               blur: bool = False) -> Image.Image:
    """Pad an image on all sides with optional edge replication (fill)."""
    left_edge = image.crop((0, 1, 1, image.height - 1))
    right_edge = image.crop((image.width - 1, 1, image.width, image.height - 1))
    top_edge = image.crop((1, 0, image.width - 1, 1))
    bottom_edge = image.crop((1, image.height - 1, image.width - 1, image.height))
    new_width = image.width + left_pad + right_pad
    new_height = image.height + top_pad + bottom_pad
    padded_image = Image.new(image.mode, (new_width, new_height))
    padded_image.paste(image, (left_pad, top_pad))
    if fill:
        if left_pad > 0:
            padded_image.paste(left_edge.resize((left_pad, new_height), resample=Image.Resampling.NEAREST), (0, 0))
        if right_pad > 0:
            padded_image.paste(right_edge.resize((right_pad, new_height),
                               resample=Image.Resampling.NEAREST), (new_width - right_pad, 0))
        if top_pad > 0:
            padded_image.paste(top_edge.resize((new_width, top_pad), resample=Image.Resampling.NEAREST), (0, 0))
        if bottom_pad > 0:
            padded_image.paste(bottom_edge.resize((new_width, bottom_pad),
                               resample=Image.Resampling.NEAREST), (0, new_height - bottom_pad))
    return padded_image


def resize_and_pad_image(image: Image.Image, width: int, height: int,
                         fill: bool = False, blur: bool = False) -> tuple:
    """Resize image to fit target bounds, then pad borders."""
    width_ratio = width / image.width
    height_ratio = height / image.height
    resize_ratio = min(width_ratio, height_ratio)
    resize_width = round(image.width * resize_ratio)
    resize_height = round(image.height * resize_ratio)
    resized = image.resize((resize_width, resize_height), resample=Image.Resampling.LANCZOS)
    horizontal_pad = (width - resize_width) // 2
    vertical_pad = (height - resize_height) // 2
    result = pad_image2(resized, horizontal_pad, horizontal_pad, vertical_pad, vertical_pad, fill, blur)
    result = result.resize((width, height), resample=Image.Resampling.LANCZOS)
    return result, (horizontal_pad, vertical_pad)


def region_intersection(region1: tuple, region2: tuple) -> tuple:
    """Find intersection bounding box of two regions, or None if they do not intersect."""
    x1, y1, x2, y2 = region1
    x1_, y1_, x2_, y2_ = region2
    x1 = max(x1, x1_)
    y1 = max(y1, y1_)
    x2 = min(x2, x2_)
    y2 = min(y2, y2_)
    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)


def crop_controlnet(cond_dict: dict, regions: list, init_size: tuple,
                    canvas_size: tuple, tile_size: tuple, w_pad: int, h_pad: int):
    """Crop and resize controlnet hint images for the tile."""
    if "control" not in cond_dict:
        return
    if not isinstance(regions, list):
        regions = [regions]
    c = cond_dict["control"]
    controlnet = c.copy()
    cond_dict["control"] = controlnet
    while c is not None:
        hint = controlnet.cond_hint_original
        tiled_hints = []
        for region in regions:
            resized_crop = resize_region(region, canvas_size, hint.shape[:-3:-1])
            tiled_hint = crop_tensor(hint.movedim(1, -1), resized_crop).movedim(-1, 1)
            tiled_hint = resize_tensor(tiled_hint, tile_size[::-1])
            tiled_hints.append(tiled_hint)
        controlnet.cond_hint_original = torch.cat(tiled_hints, dim=0)
        c = c.previous_controlnet
        controlnet.set_previous_controlnet(c.copy() if c is not None else None)
        controlnet = controlnet.previous_controlnet


def crop_gligen(cond_dict: dict, regions: list, init_size: tuple,
                canvas_size: tuple, tile_size: tuple, w_pad: int, h_pad: int):
    """Crop and offset GLIGEN bounding box coordinates for the tile."""
    if "gligen" not in cond_dict:
        return
    region = regions if isinstance(regions, tuple) else regions[0]
    type, model, cond = cond_dict["gligen"]
    if type != "position":
        return
    cropped = []
    for c in cond:
        emb, h, w, y, x = c
        x1 = x * 8
        y1 = y * 8
        x2 = x1 + w * 8
        y2 = y1 + h * 8
        gligen_upscaled_box = resize_region((x1, y1, x2, y2), init_size, canvas_size)
        intersection = region_intersection(gligen_upscaled_box, region)
        if intersection is None:
            continue
        x1, y1, x2, y2 = intersection
        x1 -= region[0]
        y1 -= region[1]
        x2 -= region[0]
        y2 -= region[1]
        x1 += w_pad
        y1 += h_pad
        x2 += w_pad
        y2 += h_pad
        h = (y2 - y1) // 8
        w = (x2 - x1) // 8
        x = x1 // 8
        y = y1 // 8
        cropped.append((emb, h, w, y, x))
    cond_dict["gligen"] = (type, model, cropped)


def crop_area(cond_dict: dict, regions: list, init_size: tuple,
              canvas_size: tuple, tile_size: tuple, w_pad: int, h_pad: int):
    """Crop and offset conditioning area regions for the tile."""
    if "area" not in cond_dict:
        return
    region = regions if isinstance(regions, tuple) else regions[0]
    h, w, y, x = cond_dict["area"]
    w, h, x, y = 8 * w, 8 * h, 8 * x, 8 * y
    x1, y1, x2, y2 = resize_region((x, y, x + w, y + h), init_size, canvas_size)
    intersection = region_intersection((x1, y1, x2, y2), region)
    if intersection is None:
        del cond_dict["area"]
        del cond_dict["strength"]
        return
    x1, y1, x2, y2 = intersection
    x1 -= region[0]
    y1 -= region[1]
    x2 -= region[0]
    y2 -= region[1]
    x1 += w_pad
    y1 += h_pad
    x2 += w_pad
    y2 += h_pad
    w, h = (x2 - x1) // 8, (y2 - y1) // 8
    x, y = x1 // 8, y1 // 8
    cond_dict["area"] = (h, w, y, x)


def crop_mask(cond_dict: dict, regions: list, init_size: tuple,
              canvas_size: tuple, tile_size: tuple, w_pad: int, h_pad: int):
    """Crop and offset conditioning masks for the tile."""
    if "mask" not in cond_dict:
        return
    region = regions if isinstance(regions, tuple) else regions[0]
    mask_tensor = cond_dict["mask"]
    masks = []
    for i in range(mask_tensor.shape[0]):
        mask = tensor_to_pil(mask_tensor, i)
        mask = mask.resize(canvas_size, Image.Resampling.BICUBIC)
        mask = mask.crop(region)
        mask, _ = resize_and_pad_image(mask, tile_size[0], tile_size[1], fill=True)
        if tile_size != mask.size:
            mask = mask.resize(tile_size, Image.Resampling.BICUBIC)
        mask = pil_to_tensor(mask)
        mask = mask.squeeze(-1)
        masks.append(mask)
    cond_dict["mask"] = torch.cat(masks, dim=0)


def crop_reference_latents(cond_dict: dict, regions: list, init_size: tuple,
                           canvas_size: tuple, tile_size: tuple, w_pad: int,
                           h_pad: int, divisor: int = 8):
    """Crop and resize reference latents for the tile."""
    latents = cond_dict.get("reference_latents")
    if not isinstance(latents, list):
        return
    region = regions if isinstance(regions, tuple) else regions[0]
    k = divisor
    W_can_px, H_can_px = canvas_size
    W_can_lat, H_can_lat = W_can_px // k, H_can_px // k
    W_tile_px, H_tile_px = tile_size
    W_tile_lat, H_tile_lat = max(1, W_tile_px // k), max(1, H_tile_px // k)
    x1_px, y1_px, x2_px, y2_px = region
    new_latents = []
    for t in latents:
        has_5d = False
        if t.ndim == 5:
            has_5d = True
            t = t.squeeze(2)
        if t.ndim != 4:
            raise ValueError(f"expected BCHW, got {t.shape}")
        if t.shape[-2:] != (H_can_lat, W_can_lat):
            t = F.interpolate(t, size=(H_can_lat, W_can_lat), mode="bilinear", align_corners=False)
        H_lat_max, W_lat_max = t.shape[-2], t.shape[-1]
        w0_lat = max(0, min(W_lat_max, int(round(x1_px / k))))
        w1_lat = max(0, min(W_lat_max, int(round(x2_px / k))))
        h0_lat = max(0, min(H_lat_max, int(round(y1_px / k))))
        h1_lat = max(0, min(H_lat_max, int(round(y2_px / k))))
        
        # Ensure we crop at least 1x1 if coordinates collapse
        if w0_lat >= w1_lat:
            if w0_lat > 0:
                w0_lat -= 1
            else:
                w1_lat = min(W_lat_max, w0_lat + 1)
        if h0_lat >= h1_lat:
            if h0_lat > 0:
                h0_lat -= 1
            else:
                h1_lat = min(H_lat_max, h0_lat + 1)

        cropped = t[:, :, h0_lat:h1_lat, w0_lat:w1_lat]
        cropped = F.interpolate(cropped, size=(H_tile_lat, W_tile_lat), mode="bilinear", align_corners=False)
        if has_5d:
            cropped = cropped.unsqueeze(2)
        new_latents.append(cropped)
    cond_dict["reference_latents"] = new_latents


def crop_cond(cond: list, regions: list, init_size: tuple, canvas_size: tuple,
              tile_size: tuple, w_pad: int = 0, h_pad: int = 0,
              divisor: int = 8) -> list:
    """Crop all conditioning features (ControlNet, GLIGEN, area masks, etc.) to target tile."""
    cropped = []
    for emb, x in cond:
        cond_dict = x.copy()
        n = [emb, cond_dict]
        crop_controlnet(cond_dict, regions, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_gligen(cond_dict, regions, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_area(cond_dict, regions, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_mask(cond_dict, regions, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_reference_latents(cond_dict, regions, init_size, canvas_size, tile_size, w_pad, h_pad, divisor)
        cropped.append(n)
    return cropped


# ---------------------------------------------------------------------------
# Advanced Masking
# ---------------------------------------------------------------------------

def create_smooth_matrix_mask(canvas_w: int, canvas_h: int, core_x1: int,
                              core_y1: int, core_x2: int, core_y2: int,
                              blend: int) -> Image.Image:
    """Generate a smooth blending mask for composition to avoid hard edges."""
    mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    x1 = max(0, core_x1 - blend)
    y1 = max(0, core_y1 - blend)
    x2 = min(canvas_w, core_x2 + blend)
    y2 = min(canvas_h, core_y2 + blend)
    if x1 >= x2 or y1 >= y2:
        return Image.fromarray((mask * 255).astype(np.uint8), mode='L')

    def smooth_curve(length):
        t = np.linspace(0, 1, length, dtype=np.float32)
        return 0.5 - 0.5 * np.cos(np.pi * t)

    x_grad = np.ones(x2 - x1, dtype=np.float32)
    y_grad = np.ones(y2 - y1, dtype=np.float32)

    blend_left = core_x1 - x1
    blend_right = x2 - core_x2
    blend_top = core_y1 - y1
    blend_bottom = y2 - core_y2

    if blend_left > 0:
        x_grad[:blend_left] = smooth_curve(blend_left)
    if blend_right > 0:
        x_grad[-blend_right:] = smooth_curve(blend_right)[::-1]
    if blend_top > 0:
        y_grad[:blend_top] = smooth_curve(blend_top)
    if blend_bottom > 0:
        y_grad[-blend_bottom:] = smooth_curve(blend_bottom)[::-1]

    mask_2d = np.outer(y_grad, x_grad)
    mask[y1:y2, x1:x2] = mask_2d
    return Image.fromarray((mask * 255).astype(np.uint8), mode='L')


# ---------------------------------------------------------------------------
# Tile preparation
# ---------------------------------------------------------------------------

def expand_and_align_crop(region: tuple, width: int, height: int,
                         target_w: int, target_h: int, pixel_align: int) -> tuple:
    """Expand crop region to target size, centered, aligned to pixel_align."""
    x1, y1, x2, y2 = region
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    new_x1 = cx - (target_w // 2)
    new_y1 = cy - (target_h // 2)

    # Align to model-compatible boundaries
    new_x1 = (new_x1 // pixel_align) * pixel_align
    new_y1 = (new_y1 // pixel_align) * pixel_align
    new_x2 = new_x1 + target_w
    new_y2 = new_y1 + target_h

    # Clamp to canvas, re-align robustly
    if new_x1 < 0:
        new_x1 = 0
        new_x2 = min(width, target_w)
    if new_y1 < 0:
        new_y1 = 0
        new_y2 = min(height, target_h)
    if new_x2 > width:
        new_x2 = width
        new_x1 = max(0, width - target_w)
    if new_y2 > height:
        new_y2 = height
        new_y1 = max(0, height - target_h)

    new_x1 = max(0, (new_x1 // pixel_align) * pixel_align)
    new_y1 = max(0, (new_y1 // pixel_align) * pixel_align)
    new_x2 = min(width, new_x1 + target_w)
    new_y2 = min(height, new_y1 + target_h)
    return (new_x1, new_y1, new_x2, new_y2), (target_w, target_h)


def prepare_tile(image: Image.Image, core_x1: int, core_y1: int,
                 actual_tw: int, actual_th: int, padding: int,
                 canvas_w: int, canvas_h: int, full_tile_w: int = None,
                 full_tile_h: int = None, pixel_align: int = 16) -> tuple:
    """Crop a tile region from the canvas with padding, aligned to pixel_align."""
    use_tw = full_tile_w if full_tile_w is not None else actual_tw
    use_th = full_tile_h if full_tile_h is not None else actual_th
    target_w = max(pixel_align, math.ceil((use_tw + padding * 2) / pixel_align) * pixel_align)
    target_h = max(pixel_align, math.ceil((use_th + padding * 2) / pixel_align) * pixel_align)

    init_x1 = max(0, core_x1 - padding)
    init_y1 = max(0, core_y1 - padding)
    init_x2 = min(canvas_w, core_x1 + actual_tw + padding)
    init_y2 = min(canvas_h, core_y1 + actual_th + padding)
    init_region = (init_x1, init_y1, init_x2, init_y2)

    crop_region, tile_size = expand_and_align_crop(
        init_region, canvas_w, canvas_h, target_w, target_h, pixel_align
    )
    tile = image.crop(crop_region)
    original_size = tile.size

    if tile.size != tile_size:
        tile = tile.resize(tile_size, Image.Resampling.LANCZOS)
    return tile, crop_region, tile_size, original_size


def composite_tile(canvas: Image.Image, tile_pil: Image.Image,
                   crop_region: tuple, original_size: tuple,
                   mask: Image.Image) -> Image.Image:
    """Paste processed tile back to canvas using blending mask."""
    if tile_pil.size != original_size:
        tile_pil = tile_pil.resize(original_size, Image.Resampling.LANCZOS)

    tile_mask = mask.crop((crop_region[0], crop_region[1], crop_region[2], crop_region[3]))
    canvas.paste(tile_pil, crop_region[:2], tile_mask.convert('L'))
    return canvas


# ---------------------------------------------------------------------------
# Noise handling
# ---------------------------------------------------------------------------

def crop_noise_for_tile(global_noise: torch.Tensor, crop_region: tuple,
                        divisor: int, target_size: tuple = None) -> torch.Tensor:
    """
    Slice global noise for this tile and normalize to zero mean / unit variance.
    Handles both 4D (B,C,H,W) and 5D (B,C,T,H,W) noise tensors.
    """
    x1, y1, x2, y2 = crop_region

    rx1 = x1 // divisor
    ry1 = y1 // divisor
    rx2 = x2 // divisor
    ry2 = y2 // divisor

    if target_size is not None:
        enc_w, enc_h = target_size
    else:
        enc_w = max(1, (x2 - x1) // divisor)
        enc_h = max(1, (y2 - y1) // divisor)

    if global_noise.ndim == 5:
        tile_noise = global_noise[:, :, :, ry1:ry2, rx1:rx2].clone()
        B, C, T, H, W = tile_noise.shape
        if H != enc_h or W != enc_w:
            tile_noise = tile_noise.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            tile_noise = F.interpolate(
                tile_noise.float(), size=(enc_h, enc_w),
                mode='bilinear', align_corners=False
            ).to(global_noise.dtype)
            tile_noise = tile_noise.reshape(B, T, C, enc_h, enc_w).permute(0, 2, 1, 3, 4)
    else:
        tile_noise = global_noise[:, :, ry1:ry2, rx1:rx2].clone()
        if tile_noise.shape[2] != enc_h or tile_noise.shape[3] != enc_w:
            tile_noise = F.interpolate(
                tile_noise.float(), size=(enc_h, enc_w),
                mode='bilinear', align_corners=False
            ).to(global_noise.dtype)

    mean = tile_noise.mean()
    std = tile_noise.std()
    if std > 1e-6:
        tile_noise = (tile_noise - mean) / std

    return tile_noise


# ---------------------------------------------------------------------------
# RoPE patching for Anima (MiniTrainDIT / VideoRopePosition3DEmb)
# ---------------------------------------------------------------------------

def patch_anima_rope(pos_embedder, shift_x: int, shift_y: int):
    """
    Wrap pos_embedder generate_embeddings to offset positions for tiled sampling.
    """
    if shift_x == 0 and shift_y == 0:
        return None

    original_generate_embeddings = pos_embedder.generate_embeddings

    def patched_generate_embeddings(B_T_H_W_C, fps=None, h_ntk_factor=None,
                                     w_ntk_factor=None, t_ntk_factor=None,
                                     device=None, dtype=None):
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else pos_embedder.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else pos_embedder.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else pos_embedder.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** pos_embedder.dim_spatial_range.to(device=device))
        w_spatial_freqs = 1.0 / (w_theta ** pos_embedder.dim_spatial_range.to(device=device))
        temporal_freqs = 1.0 / (t_theta ** pos_embedder.dim_temporal_range.to(device=device))

        B, T, H, W, _ = B_T_H_W_C
        seq_len = max(H + shift_y, W + shift_x, T)
        seq = torch.arange(seq_len, dtype=torch.float, device=device)

        uniform_fps = (fps is None) or isinstance(fps, (int, float)) or (fps.min() == fps.max())
        assert (
            uniform_fps or B == 1 or T == 1
        ), "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"

        h_positions = seq[shift_y:shift_y + H]
        w_positions = seq[shift_x:shift_x + W]

        half_emb_h = torch.outer(h_positions, h_spatial_freqs)
        half_emb_w = torch.outer(w_positions, w_spatial_freqs)

        if fps is None or pos_embedder.enable_fps_modulation is False:
            half_emb_t = torch.outer(seq[:T], temporal_freqs)
        else:
            half_emb_t = torch.outer(seq[:T] / fps * pos_embedder.base_fps, temporal_freqs)

        half_emb_h = torch.stack([torch.cos(half_emb_h), -torch.sin(half_emb_h),
                                   torch.sin(half_emb_h), torch.cos(half_emb_h)], dim=-1)
        half_emb_w = torch.stack([torch.cos(half_emb_w), -torch.sin(half_emb_w),
                                   torch.sin(half_emb_w), torch.cos(half_emb_w)], dim=-1)
        half_emb_t = torch.stack([torch.cos(half_emb_t), -torch.sin(half_emb_t),
                                   torch.sin(half_emb_t), torch.cos(half_emb_t)], dim=-1)

        from einops import repeat, rearrange
        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d x -> t h w d x", h=H, w=W),
                repeat(half_emb_h, "h d x -> t h w d x", t=T, w=W),
                repeat(half_emb_w, "w d x -> t h w d x", t=T, h=H),
            ],
            dim=-2,
        )

        return rearrange(em_T_H_W_D, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()

    pos_embedder.generate_embeddings = patched_generate_embeddings
    return original_generate_embeddings


# ---------------------------------------------------------------------------
# Sampling Wrapper
# ---------------------------------------------------------------------------

def sample_tile(model, positive, negative, sampler_name: str, scheduler: str,
                steps: int, cfg: float, latent: dict, seed: int, denoise: float,
                callback=None, disable_pbar=True, disable_noise=False,
                start_step=None, last_step=None, force_full_denoise=False) -> dict:
    """Wrapper calling ComfyUI comfy.sample.sample directly for a tile with custom callback/pbar support."""
    import latent_preview
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image, latent.get("downscale_ratio_spacial", None), latent.get("downscale_ratio_temporal", None))

    batch_inds = latent["batch_index"] if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    preview_callback = latent_preview.prepare_callback(model, steps)
    
    def wrapped_callback(step, x0, x, total_steps):
        if preview_callback is not None:
            preview_callback(step, x0, x, total_steps)
        if callback is not None:
            callback(step, x0, x, total_steps)

    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=wrapped_callback, disable_pbar=disable_pbar, seed=seed)
    
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


# ---------------------------------------------------------------------------
# Architecture detection helpers
# ---------------------------------------------------------------------------

def get_vae_spatial_compression(vae) -> int:
    """Get the VAE's spatial compression ratio (pixels per latent unit)."""
    try:
        return vae.spacial_compression_decode()
    except Exception:
        return 8  # Wan21 default


def get_model_patch_spatial(model) -> int:
    """Get the diffusion model's spatial patch size."""
    try:
        return model.model.diffusion_model.patch_spatial
    except AttributeError:
        return 2  # Anima/MiniTrainDIT default


def get_pixel_alignment(vae, model) -> int:
    """Compute minimum pixel alignment = VAE_spatial * patch_spatial."""
    vae_spatial = get_vae_spatial_compression(vae)
    patch_spatial = get_model_patch_spatial(model)
    return vae_spatial * patch_spatial

