"""
Anima Tiled Spatial Upscaler - Tiled upscaling for Anima (Cosmos-derived anime model).
Features smooth single-pass matrix blending for seamless tile compositing.
"""

import math
import gc
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn.functional as F

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.conds
import comfy.model_management
from nodes import VAEEncode, VAEDecode, VAEDecodeTiled, common_ksampler


# ---------------------------------------------------------------------------
# Tensor / PIL conversions
# ---------------------------------------------------------------------------

def pil_to_tensor(image):
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def tensor_to_pil(tensor, index=0):
    arr = tensor[index].cpu().numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Conditioning Cropping Helpers
# ---------------------------------------------------------------------------

def create_latent_blend_mask(tile_h, tile_w, lcx1, lcy1, lcx2, lcy2, blend_size, device):
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


# ---------------------------------------------------------------------------
# Conditioning Cropping Helpers
# ---------------------------------------------------------------------------

def crop_tensor(tensor, region):
    x1, y1, x2, y2 = region
    return tensor[:, y1:y2, x1:x2, :]

def resize_tensor(tensor, size, mode="nearest-exact"):
    return F.interpolate(tensor, size=size, mode=mode)

def resize_region(region, init_size, resize_size):
    x1, y1, x2, y2 = region
    init_width, init_height = init_size
    resize_width, resize_height = resize_size
    x1 = math.floor(x1 * resize_width / init_width)
    x2 = math.ceil(x2 * resize_width / init_width)
    y1 = math.floor(y1 * resize_height / init_height)
    y2 = math.ceil(y2 * resize_height / init_height)
    return (x1, y1, x2, y2)

def pad_image2(image, left_pad, right_pad, top_pad, bottom_pad, fill=False, blur=False):
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

def resize_and_pad_image(image, width, height, fill=False, blur=False):
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

def region_intersection(region1, region2):
    x1, y1, x2, y2 = region1
    x1_, y1_, x2_, y2_ = region2
    x1 = max(x1, x1_)
    y1 = max(y1, y1_)
    x2 = min(x2, x2_)
    y2 = min(y2, y2_)
    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)

def crop_controlnet(cond_dict, regions, init_size, canvas_size, tile_size, w_pad, h_pad):
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

def crop_gligen(cond_dict, regions, init_size, canvas_size, tile_size, w_pad, h_pad):
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

def crop_area(cond_dict, regions, init_size, canvas_size, tile_size, w_pad, h_pad):
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

def crop_mask(cond_dict, regions, init_size, canvas_size, tile_size, w_pad, h_pad):
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

def crop_reference_latents(cond_dict, regions, init_size, canvas_size, tile_size, w_pad, h_pad, divisor=8):
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
        w0_lat = int(round(x1_px / k))
        w1_lat = int(round(x2_px / k))
        h0_lat = int(round(y1_px / k))
        h1_lat = int(round(y2_px / k))
        cropped = t[:, :, h0_lat:h1_lat, w0_lat:w1_lat]
        cropped = F.interpolate(cropped, size=(H_tile_lat, W_tile_lat), mode="bilinear", align_corners=False)
        if has_5d:
            cropped = cropped.unsqueeze(2)
        new_latents.append(cropped)
    cond_dict["reference_latents"] = new_latents

def crop_cond(cond, regions, init_size, canvas_size, tile_size, w_pad=0, h_pad=0, divisor=8):
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

def create_smooth_matrix_mask(canvas_w, canvas_h, core_x1, core_y1, core_x2, core_y2, blend):
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

def expand_and_align_crop(region, width, height, target_w, target_h, pixel_align):
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

    # Clamp to canvas, re-align
    if new_x1 < 0:
        new_x1 = 0
        new_x2 = target_w
    if new_y1 < 0:
        new_y1 = 0
        new_y2 = target_h
    if new_x2 > width:
        new_x2 = width
        new_x1 = width - target_w
    if new_y2 > height:
        new_y2 = height
        new_y1 = height - target_h

    new_x1 = (new_x1 // pixel_align) * pixel_align
    new_y1 = (new_y1 // pixel_align) * pixel_align
    new_x2 = new_x1 + target_w
    new_y2 = new_y1 + target_h
    return (new_x1, new_y1, new_x2, new_y2), (target_w, target_h)


def prepare_tile(image, core_x1, core_y1, actual_tw, actual_th, padding,
                 canvas_w, canvas_h, full_tile_w=None, full_tile_h=None,
                 pixel_align=16):
    """Crop a tile region from the canvas with padding, aligned to pixel_align."""
    # Use full_tile_w/h if provided to force identical crop sizes
    use_tw = full_tile_w if full_tile_w is not None else actual_tw
    use_th = full_tile_h if full_tile_h is not None else actual_th
    target_w = max(pixel_align, math.ceil((use_tw + padding * 2) / pixel_align) * pixel_align)
    target_h = max(pixel_align, math.ceil((use_th + padding * 2) / pixel_align) * pixel_align)

    # Compute initial crop region from core + padding
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


def composite_tile(canvas, tile_pil, crop_region, original_size, mask):
    if tile_pil.size != original_size:
        tile_pil = tile_pil.resize(original_size, Image.Resampling.LANCZOS)

    tile_mask = mask.crop((crop_region[0], crop_region[1], crop_region[2], crop_region[3]))
    canvas.paste(tile_pil, crop_region[:2], tile_mask.convert('L'))
    return canvas


# ---------------------------------------------------------------------------
# Noise handling
# ---------------------------------------------------------------------------

def crop_noise_for_tile(global_noise, crop_region, divisor):
    """
    Slice global noise for this tile and normalize to zero mean / unit variance.
    Handles both 4D (B,C,H,W) and 5D (B,C,T,H,W) noise tensors.
    """
    x1, y1, x2, y2 = crop_region

    rx1 = x1 // divisor
    ry1 = y1 // divisor
    rx2 = x2 // divisor
    ry2 = y2 // divisor

    enc_w = max(1, (x2 - x1) // divisor)
    enc_h = max(1, (y2 - y1) // divisor)

    if global_noise.ndim == 5:
        # (B, C, T, H, W) — Wan21 / Cosmos 5D latent format
        tile_noise = global_noise[:, :, :, ry1:ry2, rx1:rx2].clone()
        B, C, T, H, W = tile_noise.shape
        if H != enc_h or W != enc_w:
            # Reshape: merge B and T for spatial interpolation, then restore
            tile_noise = tile_noise.reshape(B * T, C, H, W)
            tile_noise = F.interpolate(
                tile_noise.float(), size=(enc_h, enc_w),
                mode='bilinear', align_corners=False
            ).to(global_noise.dtype)
            tile_noise = tile_noise.reshape(B, T, C, enc_h, enc_w).permute(0, 2, 1, 3, 4)
    else:
        # (B, C, H, W) — standard 4D latent format
        tile_noise = global_noise[:, :, ry1:ry2, rx1:rx2].clone()
        if tile_noise.shape[2] != enc_h or tile_noise.shape[3] != enc_w:
            tile_noise = F.interpolate(
                tile_noise.float(), size=(enc_h, enc_w),
                mode='bilinear', align_corners=False
            ).to(global_noise.dtype)

    # Normalize to unit gaussian to prevent texture unevenness across tiles
    mean = tile_noise.mean()
    std = tile_noise.std()
    if std > 1e-6:
        tile_noise = (tile_noise - mean) / std

    return tile_noise


# ---------------------------------------------------------------------------
# RoPE patching for Anima (MiniTrainDIT / VideoRopePosition3DEmb)
# ---------------------------------------------------------------------------

def patch_anima_rope(pos_embedder, shift_x, shift_y):
    """
    Thin monkey-patch that offsets RoPE position indices for tiled generation.
    Instead of reimplementing the entire generate_embeddings, we wrap the
    original to inject position offsets into the sequence indices.
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

        # Correct seq length: needs to cover max(H + shift_y, W + shift_x, T)
        seq_len = max(H + shift_y, W + shift_x, T)
        seq = torch.arange(seq_len, dtype=torch.float, device=device)

        uniform_fps = (fps is None) or isinstance(fps, (int, float)) or (fps.min() == fps.max())
        assert (
            uniform_fps or B == 1 or T == 1
        ), "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"

        # Apply position offsets: shifted indices for H and W dimensions
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
# Sampling
# ---------------------------------------------------------------------------

def sample_tile(model, positive, negative, sampler_name, scheduler, steps, cfg, latent, seed, denoise):
    (samples,) = common_ksampler(
        model, seed, steps, cfg, sampler_name,
        scheduler, positive, negative, latent, denoise=denoise
    )
    return samples


# ---------------------------------------------------------------------------
# Architecture detection helpers
# ---------------------------------------------------------------------------

def get_vae_spatial_compression(vae):
    """Get the VAE's spatial compression ratio (pixels per latent unit)."""
    try:
        return vae.spacial_compression_decode()
    except Exception:
        return 8  # Wan21 default

def get_model_patch_spatial(model):
    """Get the diffusion model's spatial patch size."""
    try:
        return model.model.diffusion_model.patch_spatial
    except AttributeError:
        return 2  # Anima/MiniTrainDIT default

def get_pixel_alignment(vae, model):
    """Compute minimum pixel alignment = VAE_spatial * patch_spatial."""
    vae_spatial = get_vae_spatial_compression(vae)
    patch_spatial = get_model_patch_spatial(model)
    return vae_spatial * patch_spatial


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class AnimaTiledUpscalerNode:
    TILING_STRATEGIES = ["Chess", "Linear", "Reverse Chess", "Spiral", "Detail-First"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":         ("MODEL",),
                "positive":      ("CONDITIONING",),
                "negative":      ("CONDITIONING",),
                "vae":           ("VAE",),
                "image":         ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "seed":          ("INT",    {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps":         ("INT",    {"default": 20, "min": 1, "max": 10000}),
                "cfg":           ("FLOAT",  {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler_name":  (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler":     (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise":       ("FLOAT",  {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale_factor":  ("FLOAT",  {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.25}),
                "tiling_strategy": (cls.TILING_STRATEGIES, {
                    "default": "Detail-First",
                    "tooltip": "Processing order of the tiles. 'Detail-First' generates sharp foreground structures first, allowing smooth areas to cleanly anchor to them later."
                }),
                "tile_size_mode": (["Auto", "Adaptive (Quadtree)", "Manual"], {
                    "default": "Auto",
                    "tooltip": "'Auto' dynamically divides your image into symmetrical, equal-sized tiles near the target size. 'Adaptive (Quadtree)' splits high-detail regions into smaller tiles. 'Manual' uses your exact custom width/height."
                }),
                "target_tile_size": ("INT", {
                    "default": 1024, "min": 256, "max": 2048, "step": 64,
                    "tooltip": "Target/maximum tile size in pixels. Used by 'Auto' and 'Adaptive (Quadtree)' modes."
                }),
                "min_tile_size": ("INT", {
                    "default": 512, "min": 256, "max": 1024, "step": 64,
                    "tooltip": "Minimum tile size in pixels for 'Adaptive (Quadtree)' mode. Regions with detail higher than the threshold will be split down to this size."
                }),
                "tile_width":    ("INT",    {
                    "default": 1024, "min": 512, "max": 4096, "step": 64,
                    "tooltip": "Manual tile width in pixels. Only used when Tile Size Mode is set to 'Manual'."
                }),
                "tile_height":   ("INT",    {
                    "default": 1024, "min": 512, "max": 4096, "step": 64,
                    "tooltip": "Manual tile height in pixels. Only used when Tile Size Mode is set to 'Manual'."
                }),
                "padding":       ("INT",    {
                    "default": 128, "min": 0, "max": 512, "step": 64,
                    "tooltip": "Overlapping context margin (in pixels) on all four sides of each tile to ensure smooth transition alignment."
                }),
                "mask_blur":     ("INT",    {
                    "default": 32, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Applies a smooth Gaussian feathering to the visual seam-blending mask to merge adjacent tile transitions."
                }),
                "adaptive_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Saves render time by dynamically skipping flat tiles entirely or reducing denoising steps on medium-detail tiles."
                }),
                "tiled_decode":  ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Decodes the combined latent canvas in tiles. Crucial to prevent Out-Of-Memory (OOM) errors when outputting extremely high resolutions (8k+)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "FCN/sampling"

    def upscale(self, model, positive, negative, vae, image, upscale_model,
                seed, steps, cfg, sampler_name, scheduler, denoise, scale_factor,
                tiling_strategy, tile_size_mode, target_tile_size, min_tile_size, tile_width, tile_height, padding,
                mask_blur, adaptive_tiling, tiled_decode):

        import comfy_extras.nodes_upscale_model as upscale_nodes

        vae_encoder = VAEEncode()
        vae_decoder = VAEDecode()
        vae_decoder_tiled = VAEDecodeTiled()
        upscaler_node = upscale_nodes.ImageUpscaleWithModel()

        # --- Derive architecture constants from the actual model/VAE ---
        vae_spatial = get_vae_spatial_compression(vae)
        pixel_align = get_pixel_alignment(vae, model)
        patch_spatial = get_model_patch_spatial(model)

        print(f"[ANIMA] VAE spatial compression: {vae_spatial}x | "
              f"Patch spatial: {patch_spatial} | Pixel align: {pixel_align}px")

        batch_size = image.shape[0]
        final_batch_outputs = []

        # Save the original KSampler execution method for noise injection
        current_sample_method = comfy.samplers.KSampler.sample
        current_tile_noise = None

        def native_comfy_sample(self, noise, *args, **kwargs):
            injected_noise = current_tile_noise if current_tile_noise is not None else noise
            return current_sample_method(self, injected_noise, *args, **kwargs)

        pos_embedder_to_restore = None
        orig_pos_embedder_forward = None

        try:
            # Bypass legacy Tiled Diffusion monkeypatch
            comfy.samplers.KSampler.sample = native_comfy_sample

            for b in range(batch_size):
                print(f"[ANIMA] Processing batch element {b+1}/{batch_size}")

                img_b = image[b:b+1]  # (1, H, W, C)
                b_w, b_h = img_b.shape[2], img_b.shape[1]
                print(f"[ANIMA] Input: {b_w}x{b_h}")

                # --- Model upscale + snap to alignment ---
                upscaled_t = upscaler_node.upscale(upscale_model=upscale_model, image=img_b)[0]

                target_w = (round(b_w * scale_factor) // pixel_align) * pixel_align
                target_h = (round(b_h * scale_factor) // pixel_align) * pixel_align

                upscaled_t = upscaled_t.movedim(-1, 1)
                orig_dtype = upscaled_t.dtype
                upscaled_t = F.interpolate(
                    upscaled_t.float(),
                    size=(target_h, target_w),
                    mode='bicubic',
                    antialias=True
                ).to(orig_dtype).movedim(1, -1)

                canvas_w, canvas_h = upscaled_t.shape[2], upscaled_t.shape[1]
                canvas_np = (upscaled_t[0].cpu().numpy() * 255).astype(np.uint8)
                canvas = Image.fromarray(canvas_np)

                print(f"[ANIMA] Canvas: {canvas_w}x{canvas_h}")
                print("[ANIMA] Encoding upscaled reference latent...")

                # --- Encode reference for noise generation ---
                (upscaled_latent_dict,) = vae_encoder.encode(vae=vae, pixels=upscaled_t)
                raw_latent = upscaled_latent_dict["samples"]

                # Dynamically compute VAE spatial divisor from actual tensor shapes
                # raw_latent is either (B,C,H,W) or (B,C,T,H,W)
                latent_w = raw_latent.shape[-1]
                noise_divisor = max(1, canvas_w // latent_w)

                print(f"[ANIMA] Noise divisor: {noise_divisor} (latent shape: {list(raw_latent.shape)})")

                global_noise = comfy.sample.prepare_noise(raw_latent, seed, None)
                latent_b = raw_latent.clone()

                # --- Compute edge variance from low-res input ---
                gray_lr = (img_b[..., 0] * 0.299 + img_b[..., 1] * 0.587 + img_b[..., 2] * 0.114).unsqueeze(1)

                blur_kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=img_b.device) / 9.0
                blur_kernel = blur_kernel.to(device=gray_lr.device, dtype=gray_lr.dtype)
                gray_lr = F.conv2d(gray_lr, blur_kernel, padding=1)

                lap_kernel = torch.tensor(
                    [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                    dtype=torch.float32, device=img_b.device
                ).view(1, 1, 3, 3).to(device=gray_lr.device, dtype=gray_lr.dtype)
                lap_map_lr = F.conv2d(gray_lr, lap_kernel, padding=1).abs()

                # Filter out low-amplitude background texture noise (e.g. grass, rock surface texture)
                # using a dynamic contrast threshold based on the image's mean and standard deviation.
                lap_mean = lap_map_lr.mean()
                lap_std = lap_map_lr.std()
                lap_threshold = lap_mean + 0.5 * lap_std
                lap_map_lr = torch.where(lap_map_lr > lap_threshold, lap_map_lr, torch.zeros_like(lap_map_lr))

                def get_region_variance(x1, y1, x2, y2):
                    lr_x1 = int(x1 / scale_factor)
                    lr_y1 = int(y1 / scale_factor)
                    lr_x2 = min(int(x2 / scale_factor), lap_map_lr.shape[3])
                    lr_y2 = min(int(y2 / scale_factor), lap_map_lr.shape[2])
                    if lr_x2 <= lr_x1 or lr_y2 <= lr_y1:
                        return 0.0
                    tile_lap = lap_map_lr[:, :, lr_y1:lr_y2, lr_x1:lr_x2]
                    return float(tile_lap.mean()) if tile_lap.numel() > 0 else 0.0

                # --- Compute tile grid ---
                if tile_size_mode == "Adaptive (Quadtree)":
                    # Get baseline variances of min_tile_size blocks to compute adaptive split_threshold and ref_var
                    baseline_variances = []
                    grid_sz = max(pixel_align, min_tile_size)
                    for gy in range(0, canvas_h, grid_sz):
                        for gx in range(0, canvas_w, grid_sz):
                            gx2 = min(gx + grid_sz, canvas_w)
                            gy2 = min(gy + grid_sz, canvas_h)
                            baseline_variances.append(get_region_variance(gx, gy, gx2, gy2))
                    
                    baseline_variances = sorted(baseline_variances)
                    if baseline_variances:
                        pct_idx = min(len(baseline_variances) - 1, max(0, int(len(baseline_variances) * 0.60)))
                        ref_var = max(0.005, baseline_variances[pct_idx])
                    else:
                        ref_var = 1.0

                    split_threshold = 0.5 * ref_var
                    raw_tiles = []

                    def partition_tile(x1, y1, x2, y2):
                        w = x2 - x1
                        h = y2 - y1
                        if w <= 0 or h <= 0:
                            return
                        
                        var = get_region_variance(x1, y1, x2, y2)
                        
                        should_split = False
                        if w > target_tile_size or h > target_tile_size:
                            should_split = True
                        elif var > split_threshold and w > min_tile_size and h > min_tile_size:
                            should_split = True
                            
                        if should_split:
                            cx = x1 + w // 2
                            cy = y1 + h // 2
                            cx = (cx // pixel_align) * pixel_align
                            cy = (cy // pixel_align) * pixel_align
                            
                            if cx <= x1 or cx >= x2:
                                cx = x1 + (w // 2)
                                cx = max(x1 + pixel_align, (cx // pixel_align) * pixel_align)
                            if cy <= y1 or cy >= y2:
                                cy = y1 + (h // 2)
                                cy = max(y1 + pixel_align, (cy // pixel_align) * pixel_align)
                                
                            partition_tile(x1, y1, cx, cy)
                            partition_tile(cx, y1, x2, cy)
                            partition_tile(x1, cy, cx, y2)
                            partition_tile(cx, cy, x2, y2)
                        else:
                            raw_tiles.append((x1, y1, w, h, var))

                    partition_tile(0, 0, canvas_w, canvas_h)
                    
                    # Convert to standard format
                    tiles_order = []
                    tile_variances = {}
                    for idx, (x1, y1, w, h, var) in enumerate(raw_tiles):
                        xi, yi = idx, 0
                        tiles_order.append((xi, yi, x1, y1, w, h))
                        tile_variances[(xi, yi)] = var

                    current_tile_width = None
                    current_tile_height = None
                    rows = None
                    cols = None
                else:
                    if tile_size_mode == "Auto":
                        cols = max(1, round(canvas_w / target_tile_size))
                        rows = max(1, round(canvas_h / target_tile_size))
                        current_tile_width = max(pixel_align * 4,
                                                 math.ceil((canvas_w / cols) / pixel_align) * pixel_align)
                        current_tile_height = max(pixel_align * 4,
                                                  math.ceil((canvas_h / rows) / pixel_align) * pixel_align)
                    else:
                        current_tile_width = (tile_width // pixel_align) * pixel_align
                        current_tile_height = (tile_height // pixel_align) * pixel_align

                    rows = math.ceil(canvas_h / current_tile_height)
                    cols = math.ceil(canvas_w / current_tile_width)
                    tiles_order = []

                    for yi in range(rows):
                        for xi in range(cols):
                            core_x1 = xi * current_tile_width
                            core_y1 = yi * current_tile_height
                            actual_tw = min(current_tile_width, canvas_w - core_x1)
                            actual_th = min(current_tile_height, canvas_h - core_y1)
                            if actual_tw > 0 and actual_th > 0:
                                tiles_order.append((xi, yi, core_x1, core_y1, actual_tw, actual_th))

                    tile_variances = {}
                    for xi, yi, core_x1, core_y1, actual_tw, actual_th in tiles_order:
                        tile_variances[(xi, yi)] = get_region_variance(core_x1, core_y1, core_x1 + actual_tw, core_y1 + actual_th)

                    var_values = sorted(list(tile_variances.values()))
                    if var_values:
                        pct_idx = min(len(var_values) - 1, max(0, int(len(var_values) * 0.60)))
                        ref_var = max(0.005, var_values[pct_idx])
                    else:
                        ref_var = 1.0

                total = len(tiles_order)

                # --- Apply tiling strategy ---
                if tiling_strategy == "Chess":
                    tiles_order = ([t for t in tiles_order if (t[0]+t[1]) % 2 == 0] +
                                   [t for t in tiles_order if (t[0]+t[1]) % 2 == 1])
                elif tiling_strategy == "Reverse Chess":
                    tiles_order = ([t for t in tiles_order if (t[0]+t[1]) % 2 == 1] +
                                   [t for t in tiles_order if (t[0]+t[1]) % 2 == 0])
                elif tiling_strategy == "Spiral":
                    if tile_size_mode == "Adaptive (Quadtree)":
                        tiles_order.sort(key=lambda t: (t[2] + t[4]/2 - canvas_w/2)**2 + (t[3] + t[5]/2 - canvas_h/2)**2)
                    else:
                        cx, cy = (cols - 1) / 2.0, (rows - 1) / 2.0
                        tiles_order.sort(key=lambda t: (t[0]-cx)**2 + (t[1]-cy)**2)
                elif tiling_strategy == "Detail-First":
                    tiles_order.sort(key=lambda t: tile_variances[(t[0], t[1])], reverse=True)

                if tile_size_mode == "Adaptive (Quadtree)":
                    print(f"[ANIMA] Grid: {total} tiles (Adaptive Quadtree, strategy: {tiling_strategy})")
                else:
                    print(f"[ANIMA] Grid: {rows}x{cols} = {total} tiles "
                          f"(tile size: {current_tile_width}x{current_tile_height}, "
                          f"mode: {tile_size_mode}, strategy: {tiling_strategy})")

                pbar = comfy.utils.ProgressBar(total)

                for step_i, (xi, yi, core_x1, core_y1, actual_tw, actual_th) in enumerate(tiles_order):
                    print(f"[ANIMA] Tile {step_i+1}/{total} ({xi},{yi}) "
                          f"core=({core_x1},{core_y1}) size={actual_tw}x{actual_th}")

                    # --- Adaptive Tiling / Skip Check ---
                    actual_denoise = denoise
                    if adaptive_tiling:
                        var_ratio = tile_variances[(xi, yi)] / ref_var
                        abs_var = tile_variances[(xi, yi)]
                        if abs_var < 0.002 or var_ratio < 0.15:  # Absolute flat threshold OR relative ratio
                            print(f"[ANIMA] Skipping flat tile ({xi},{yi}), abs_var: {abs_var:.4f}, ratio: {var_ratio:.3f}")
                            pbar.update(1)
                            comfy.model_management.throw_exception_if_processing_interrupted()
                            continue
                        # For medium-detail tiles, still reduce denoise somewhat
                        if var_ratio < 0.5:
                            actual_denoise = denoise * (0.5 + 0.5 * var_ratio)
                            print(f"[ANIMA] Adaptive denoise (medium-detail): {var_ratio:.3f} "
                                  f"(effective: {actual_denoise:.4f})")

                    if tile_size_mode == "Adaptive (Quadtree)":
                        # Ensure the tile has a minimum context window of target_tile_size (including padding)
                        # to prevent local hallucinations (like drawing a torso near the foot or a phone on the hand)
                        min_context_w = max(0, target_tile_size - padding * 2)
                        min_context_h = max(0, target_tile_size - padding * 2)
                        full_tile_w = max(actual_tw, min_context_w)
                        full_tile_h = max(actual_th, min_context_h)
                    else:
                        full_tile_w = current_tile_width
                        full_tile_h = current_tile_height

                    tile_pil, crop_region, tile_size, orig_size = prepare_tile(
                        canvas, core_x1, core_y1, actual_tw, actual_th, padding,
                        canvas_w, canvas_h, full_tile_w=full_tile_w,
                        full_tile_h=full_tile_h, pixel_align=pixel_align)

                    core_x2 = core_x1 + actual_tw
                    core_y2 = core_y1 + actual_th

                    # --- Visual blend size ---
                    visual_blend_size = max(16, padding - 64) if padding >= 64 else (padding // 2)

                    # --- Crop latent tile directly from global latent canvas ---
                    x1, y1, x2, y2 = crop_region
                    lx1 = x1 // noise_divisor
                    ly1 = y1 // noise_divisor
                    lx2 = x2 // noise_divisor
                    ly2 = y2 // noise_divisor

                    if latent_b.ndim == 5:
                        cropped_latent = latent_b[:, :, :, ly1:ly2, lx1:lx2].clone()
                        tile_h_lat, tile_w_lat = cropped_latent.shape[3], cropped_latent.shape[4]
                    else:
                        cropped_latent = latent_b[:, :, ly1:ly2, lx1:lx2].clone()
                        tile_h_lat, tile_w_lat = cropped_latent.shape[2], cropped_latent.shape[3]

                    latent = {"samples": cropped_latent}

                    # --- Inpainting noise mask ---
                    latent_expand_px = max(0, padding - 32)
                    latent_expand_size = (latent_expand_px // 32) * 32 if padding >= 32 else 0

                    l_x1_px = max(0, core_x1 - latent_expand_size)
                    l_y1_px = max(0, core_y1 - latent_expand_size)
                    l_x2_px = min(canvas_w, core_x2 + latent_expand_size)
                    l_y2_px = min(canvas_h, core_y2 + latent_expand_size)

                    # Convert pixel coords to latent coords using actual VAE divisor
                    l_x1_lat = max(0, (l_x1_px - crop_region[0]) // noise_divisor)
                    l_y1_lat = max(0, (l_y1_px - crop_region[1]) // noise_divisor)
                    l_x2_lat = min(tile_w_lat, (l_x2_px - crop_region[0]) // noise_divisor)
                    l_y2_lat = min(tile_h_lat, (l_y2_px - crop_region[1]) // noise_divisor)

                    if cropped_latent.ndim == 5:
                        # 5D mask: (1, 1, T, H, W) — T dimension is fully denoised
                        T_lat = cropped_latent.shape[2]
                        latent_mask = torch.zeros(
                            (1, T_lat, tile_h_lat, tile_w_lat),
                            dtype=torch.float32, device=cropped_latent.device
                        )
                        latent_mask[0, :, l_y1_lat:l_y2_lat, l_x1_lat:l_x2_lat] = 1.0
                    else:
                        latent_mask = torch.zeros(
                            (1, tile_h_lat, tile_w_lat),
                            dtype=torch.float32, device=cropped_latent.device
                        )
                        latent_mask[0, l_y1_lat:l_y2_lat, l_x1_lat:l_x2_lat] = 1.0

                    latent["noise_mask"] = latent_mask

                    # --- RoPE position patching for Anima ---
                    try:
                        dm = model.model.diffusion_model
                    except AttributeError:
                        dm = None

                    if dm is not None and hasattr(dm, "pos_embedder"):
                        pos_embedder_to_restore = dm.pos_embedder
                        shift_x = crop_region[0] // (noise_divisor * patch_spatial)
                        shift_y = crop_region[1] // (noise_divisor * patch_spatial)
                        orig_pos_embedder_forward = patch_anima_rope(
                            dm.pos_embedder, shift_x, shift_y
                        )

                    # --- Crop conditionings for this tile ---
                    tile_size_px = tile_size
                    cropped_pos = crop_cond(
                        positive, crop_region, (b_w, b_h),
                        (canvas_w, canvas_h), tile_size_px, divisor=noise_divisor
                    )
                    cropped_neg = crop_cond(
                        negative, crop_region, (b_w, b_h),
                        (canvas_w, canvas_h), tile_size_px, divisor=noise_divisor
                    )

                    tile_noise = crop_noise_for_tile(global_noise, crop_region, noise_divisor)

                    current_tile_noise = tile_noise

                    try:
                        sampled = sample_tile(
                            model, cropped_pos, cropped_neg, sampler_name,
                            scheduler, steps, cfg, latent, seed, actual_denoise)
                    finally:
                        current_tile_noise = None
                        if orig_pos_embedder_forward is not None:
                            pos_embedder_to_restore.generate_embeddings = orig_pos_embedder_forward
                            pos_embedder_to_restore = None
                            orig_pos_embedder_forward = None

                    # --- Composite tile in latent space ---
                    l_core_x1 = (core_x1 - crop_region[0]) // noise_divisor
                    l_core_y1 = (core_y1 - crop_region[1]) // noise_divisor
                    l_core_x2 = (core_x2 - crop_region[0]) // noise_divisor
                    l_core_y2 = (core_y2 - crop_region[1]) // noise_divisor

                    lblend = max(1, visual_blend_size // noise_divisor)

                    latent_blend_mask = create_latent_blend_mask(
                        tile_h_lat, tile_w_lat, l_core_x1, l_core_y1, l_core_x2, l_core_y2,
                        lblend, device=cropped_latent.device
                    )

                    if cropped_latent.ndim == 5:
                        latent_blend_mask = latent_blend_mask.view(1, 1, 1, tile_h_lat, tile_w_lat)
                    else:
                        latent_blend_mask = latent_blend_mask.view(1, 1, tile_h_lat, tile_w_lat)

                    sampled_samples = sampled["samples"]

                    if latent_b.ndim == 5:
                        latent_b[:, :, :, ly1:ly2, lx1:lx2] = (
                            sampled_samples * latent_blend_mask +
                            latent_b[:, :, :, ly1:ly2, lx1:lx2] * (1.0 - latent_blend_mask)
                        )
                    else:
                        latent_b[:, :, ly1:ly2, lx1:lx2] = (
                            sampled_samples * latent_blend_mask +
                            latent_b[:, :, ly1:ly2, lx1:lx2] * (1.0 - latent_blend_mask)
                        )

                    pbar.update(1)
                    comfy.model_management.throw_exception_if_processing_interrupted()

                # --- Decode global latent canvas at the end of the batch element ---
                print("[ANIMA] Decoding final latent canvas...")
                if tiled_decode:
                    (decoded,) = vae_decoder_tiled.decode(vae=vae, samples={"samples": latent_b}, tile_size=512)
                else:
                    (decoded,) = vae_decoder.decode(vae=vae, samples={"samples": latent_b})

                out_t = decoded

                final_batch_outputs.append(out_t)

                # Proactive VRAM cleanup per batch iteration
                del raw_latent, global_noise, latent_b, canvas
                gc.collect()
                comfy.model_management.soft_empty_cache()

        finally:
            comfy.samplers.KSampler.sample = current_sample_method

            if pos_embedder_to_restore is not None and orig_pos_embedder_forward is not None:
                try:
                    pos_embedder_to_restore.generate_embeddings = orig_pos_embedder_forward
                except Exception:
                    pass

        # Reconstruct standard ComfyUI output batch [B, H, W, C]
        final_out = torch.cat(final_batch_outputs, dim=0)
        return (final_out,)


# ---------------------------------------------------------------------------
# Entrypoint Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "AnimaTiledUpscaler": AnimaTiledUpscalerNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaTiledUpscaler": "Anima Tiled Upscaler",
}