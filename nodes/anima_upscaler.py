"""
Anima Tiled Spatial Upscaler - Tiled upscaling for Anima (Cosmos-derived anime model).
Features smooth single-pass matrix blending for seamless tile compositing.
"""

import math
import gc
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.conds
import comfy.model_management
import comfy_extras.nodes_upscale_model as upscale_nodes
from nodes import VAEEncode


# ---------------------------------------------------------------------------
# Import core tiled utilities
# ---------------------------------------------------------------------------

from ..core.tiled_utils import (
    pil_to_tensor, create_latent_blend_mask, crop_cond,
    patch_anima_rope, sample_tile, get_vae_spatial_compression,
    get_model_patch_spatial, get_pixel_alignment, crop_noise_for_tile
)


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
                "max_upscale_scale": ("FLOAT",  {"default": 4.0, "min": 1.0, "max": 16.0, "step": 0.25,
                    "tooltip": "The maximum scale factor to process the ROI tiles at. If the upscale model has a higher upscale factor, we downscale the reference canvas to this factor to prevent VRAM OOM."
                }),
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
                "detail_percentile": ("FLOAT", {
                    "default": 0.85, "min": 0.00, "max": 1.00, "step": 0.01,
                    "tooltip": "The quantile threshold for detail selection. A value of 0.85 means the top 15% of high frequency details are analyzed for splits (Adaptive mode) or kept (for filtering). Lower values include more of the image (less detailed areas)."
                }),
                "latent_upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"], {
                    "default": "bislerp",
                    "tooltip": "The interpolation method used to scale latent variables. 'bislerp' or 'area' is highly recommended to avoid ringing/squiggly lines near sharp edges."
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "FCN/sampling"

    def upscale(self, model, positive, negative, vae, image, upscale_model,
                seed, steps, cfg, sampler_name, scheduler, denoise, scale_factor, max_upscale_scale,
                tiling_strategy, tile_size_mode, target_tile_size, min_tile_size, tile_width, tile_height, padding,
                mask_blur, adaptive_tiling, detail_percentile, latent_upscale_method):

        vae_encoder = VAEEncode()
        upscaler_node = upscale_nodes.ImageUpscaleWithModel()

        vae_spatial = get_vae_spatial_compression(vae)
        pixel_align = get_pixel_alignment(vae, model)
        patch_spatial = get_model_patch_spatial(model)

        print(f"[ANIMA] Upscaler | VAE spatial compression: {vae_spatial}x | Patch spatial: {patch_spatial} | Pixel align: {pixel_align}px")

        batch_size = image.shape[0]
        target_latents = []

        for b in range(batch_size):
            img_b = image[b:b+1]  # (1, H, W, C)
            b_w, b_h = img_b.shape[2], img_b.shape[1]

            target_w = (round(b_w * scale_factor) // pixel_align) * pixel_align
            target_h = (round(b_h * scale_factor) // pixel_align) * pixel_align

            # --- Model upscale ---
            upscaled_t = upscaler_node.upscale(upscale_model=upscale_model, image=img_b)[0]

            # Create target canvas (2x) in pixel space
            upscaled_t_target = upscaled_t.movedim(-1, 1)
            orig_dtype = upscaled_t_target.dtype
            upscaled_t_target = F.interpolate(
                upscaled_t_target.float(), size=(target_h, target_w), mode='bicubic', antialias=True
            ).to(orig_dtype).movedim(1, -1)

            # VAE Encode target canvas once
            (target_latent_dict,) = vae_encoder.encode(vae=vae, pixels=upscaled_t_target)
            target_latents.append(target_latent_dict["samples"])

            del upscaled_t, upscaled_t_target
            gc.collect()
            comfy.model_management.soft_empty_cache()

        latent_b_batch = torch.cat(target_latents, dim=0)

        # Delegate execution to the shared execute_tiled_sampling helper
        sampled_dict = execute_tiled_sampling(
            model=model,
            positive=positive,
            negative=negative,
            latent_b_batch=latent_b_batch,
            image=image,
            vae_spatial=vae_spatial,
            patch_spatial=patch_spatial,
            pixel_align=pixel_align,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            tiling_strategy=tiling_strategy,
            tile_size_mode=tile_size_mode,
            target_tile_size=target_tile_size,
            min_tile_size=min_tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
            padding=padding,
            mask_blur=mask_blur,
            adaptive_tiling=adaptive_tiling,
            detail_percentile=detail_percentile,
            scale_factor=scale_factor
        )

        return (sampled_dict,)


class AnimaTiledSamplerNode:
    TILING_STRATEGIES = ["Chess", "Linear", "Reverse Chess", "Spiral", "Detail-First"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":         ("MODEL",),
                "positive":      ("CONDITIONING",),
                "negative":      ("CONDITIONING",),
                "latent":        ("LATENT",),
                "image":         ("IMAGE",),
                "seed":          ("INT",    {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps":         ("INT",    {"default": 20, "min": 1, "max": 10000}),
                "cfg":           ("FLOAT",  {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler_name":  (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler":     (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise":       ("FLOAT",  {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.01}),
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
                "detail_percentile": ("FLOAT", {
                    "default": 0.85, "min": 0.00, "max": 1.00, "step": 0.01,
                    "tooltip": "The quantile threshold for detail selection. A value of 0.85 means the top 15% of high frequency details are analyzed for splits (Adaptive mode) or kept (for filtering). Lower values include more of the image (less detailed areas)."
                }),
            },
            "optional": {
                "vae":           ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "FCN/sampling"

    def sample(self, model, positive, negative, latent, image,
               seed, steps, cfg, sampler_name, scheduler, denoise,
               tiling_strategy, tile_size_mode, target_tile_size, min_tile_size, tile_width, tile_height, padding,
               mask_blur, adaptive_tiling, detail_percentile, vae=None):

        vae_spatial = get_vae_spatial_compression(vae) if vae is not None else 8
        patch_spatial = get_model_patch_spatial(model)
        pixel_align = vae_spatial * patch_spatial

        print(f"[ANIMA] Tiled Sampler | VAE spatial compression: {vae_spatial}x | Patch spatial: {patch_spatial} | Pixel align: {pixel_align}px")

        latent_b_batch = latent["samples"]

        # Delegate execution to the shared execute_tiled_sampling helper
        sampled_dict = execute_tiled_sampling(
            model=model,
            positive=positive,
            negative=negative,
            latent_b_batch=latent_b_batch,
            image=image,
            vae_spatial=vae_spatial,
            patch_spatial=patch_spatial,
            pixel_align=pixel_align,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            tiling_strategy=tiling_strategy,
            tile_size_mode=tile_size_mode,
            target_tile_size=target_tile_size,
            min_tile_size=min_tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
            padding=padding,
            mask_blur=mask_blur,
            adaptive_tiling=adaptive_tiling,
            detail_percentile=detail_percentile,
            scale_factor=None
        )

        return (sampled_dict,)


from ..core.tiling_logic import (
    calculate_tiling_grid,
    draw_tiling_preview_image,
    execute_tiled_sampling_loop,
    execute_tiled_sampling
)


class AnimaTileCalculatorNode:
    TILING_STRATEGIES = ["Chess", "Linear", "Reverse Chess", "Spiral", "Detail-First"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":         ("IMAGE",),
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
                "detail_percentile": ("FLOAT", {
                    "default": 0.85, "min": 0.00, "max": 1.00, "step": 0.01,
                    "tooltip": "The quantile threshold for detail selection. A value of 0.85 means the top 15% of high frequency details are analyzed for splits (Adaptive mode) or kept (for filtering). Lower values include more of the image (less detailed areas)."
                }),
            },
            "optional": {
                "vae":           ("VAE",),
                "model":         ("MODEL",),
            }
        }

    RETURN_TYPES = ("ANIMA_TILES",)
    RETURN_NAMES = ("tiles",)
    FUNCTION = "calculate"
    CATEGORY = "FCN/sampling"

    def calculate(self, image, tiling_strategy, tile_size_mode, target_tile_size, min_tile_size,
                  tile_width, tile_height, padding, mask_blur, adaptive_tiling, detail_percentile, vae=None, model=None):
        
        vae_spatial = get_vae_spatial_compression(vae) if vae is not None else 8
        patch_spatial = get_model_patch_spatial(model) if model is not None else 2
        pixel_align = vae_spatial * patch_spatial

        print(f"[ANIMA] Tiling Calculator | VAE spatial compression: {vae_spatial}x | Patch spatial: {patch_spatial} | Pixel align: {pixel_align}px")

        # 1. Calculate the grid configs for the batch (calculator image is already upscaled if upscaling was done beforehand)
        batch_configs = calculate_tiling_grid(
            image=image,
            vae_spatial=vae_spatial,
            patch_spatial=patch_spatial,
            pixel_align=pixel_align,
            tiling_strategy=tiling_strategy,
            tile_size_mode=tile_size_mode,
            target_tile_size=target_tile_size,
            min_tile_size=min_tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
            padding=padding,
            mask_blur=mask_blur,
            adaptive_tiling=adaptive_tiling,
            detail_percentile=detail_percentile,
            target_w=image.shape[2],
            target_h=image.shape[1],
            scale_factor=1.0  # Already target size
        )

        # 2. Construct the tiles struct
        tiles_struct = {
            "batch_configs": batch_configs,
            "vae_spatial": vae_spatial,
            "patch_spatial": patch_spatial,
            "pixel_align": pixel_align,
            "padding": padding,
            "mask_blur": mask_blur,
            "adaptive_tiling": adaptive_tiling,
            "detail_percentile": detail_percentile,
            "tile_size_mode": tile_size_mode,
            "tiling_strategy": tiling_strategy,
            "target_tile_size": target_tile_size
        }

        return (tiles_struct,)


class AnimaTilePreviewNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tiles": ("ANIMA_TILES",),
                "show_tiles": ("BOOLEAN", {"default": True}),
                "show_padding": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "FCN/sampling"

    def preview(self, image, tiles, show_tiles, show_padding):
        import folder_paths
        import uuid
        import os
        from PIL import Image
        
        batch_configs = tiles["batch_configs"]
        tiling_strategy = tiles["tiling_strategy"]
        adaptive_tiling = tiles["adaptive_tiling"]
        tile_size_mode = tiles["tile_size_mode"]
        padding = tiles["padding"]
        target_tile_size = tiles["target_tile_size"]
        pixel_align = tiles["pixel_align"]
        
        drawn_image = draw_tiling_preview_image(
            image=image,
            batch_configs=batch_configs,
            tiling_strategy=tiling_strategy,
            adaptive_tiling=adaptive_tiling,
            tile_size_mode=tile_size_mode,
            padding=padding,
            target_tile_size=target_tile_size,
            pixel_align=pixel_align,
            draw_tiles=show_tiles,
            draw_padding=show_padding
        )
        
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        
        ui_images = []
        batch_size = drawn_image.shape[0]
        
        for b in range(batch_size):
            img_b = drawn_image[b]
            img_np = (img_b.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            # Generate a unique temp filename
            filename = f"anima_tile_preview_{uuid.uuid4().hex}.png"
            filepath = os.path.join(temp_dir, filename)
            img_pil.save(filepath, compress_level=4)
            
            ui_images.append({
                "filename": filename,
                "subfolder": "",
                "type": "temp"
            })
            
        return {"ui": {"images": ui_images}}




class AnimaTiledSamplerFromTilesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":         ("MODEL",),
                "positive":      ("CONDITIONING",),
                "negative":      ("CONDITIONING",),
                "latent":        ("LATENT",),
                "tiles":         ("ANIMA_TILES",),
                "seed":          ("INT",    {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps":         ("INT",    {"default": 20, "min": 1, "max": 10000}),
                "cfg":           ("FLOAT",  {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler_name":  (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler":     (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise":       ("FLOAT",  {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "vae":           ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "FCN/sampling"

    def sample(self, model, positive, negative, latent, tiles,
               seed, steps, cfg, sampler_name, scheduler, denoise, vae=None):

        # Extract tiling settings from the struct
        batch_configs = tiles["batch_configs"]
        padding = tiles["padding"]
        mask_blur = tiles["mask_blur"]
        adaptive_tiling = tiles["adaptive_tiling"]
        tile_size_mode = tiles["tile_size_mode"]
        target_tile_size = tiles["target_tile_size"]

        vae_spatial = get_vae_spatial_compression(vae) if vae is not None else tiles["vae_spatial"]
        patch_spatial = get_model_patch_spatial(model)
        pixel_align = vae_spatial * patch_spatial

        print(f"[ANIMA] Tiled Sampler From Tiles | VAE spatial: {vae_spatial}x | Patch: {patch_spatial} | Pixel align: {pixel_align}px")

        latent_b_batch = latent["samples"]
        batch_size = latent_b_batch.shape[0]

        # Inferred target canvas dimensions from latent dimensions
        target_w = latent_b_batch.shape[-1] * vae_spatial
        target_h = latent_b_batch.shape[-2] * vae_spatial

        # Construct a dummy black image representing the target canvas shape for crop_cond mapping logic
        dummy_image = torch.zeros((batch_size, target_h, target_w, 3), dtype=torch.float32, device=latent_b_batch.device)

        # Run main loop
        sampled_dict = execute_tiled_sampling_loop(
            model=model,
            positive=positive,
            negative=negative,
            latent_b_batch=latent_b_batch,
            image=dummy_image,
            vae_spatial=vae_spatial,
            patch_spatial=patch_spatial,
            pixel_align=pixel_align,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            batch_configs=batch_configs,
            padding=padding,
            mask_blur=mask_blur,
            adaptive_tiling=adaptive_tiling,
            target_tile_size=target_tile_size,
            tile_size_mode=tile_size_mode
        )

        return (sampled_dict,)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Entrypoint Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "AnimaTiledUpscaler": AnimaTiledUpscalerNode,
    "AnimaTiledSampler": AnimaTiledSamplerNode,
    "AnimaTileCalculator": AnimaTileCalculatorNode,
    "AnimaTiledSamplerFromTiles": AnimaTiledSamplerFromTilesNode,
    "AnimaTilePreview": AnimaTilePreviewNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaTiledUpscaler": "Anima Tiled Upscaler",
    "AnimaTiledSampler": "Anima Tiled Sampler",
    "AnimaTileCalculator": "Anima Tile Calculator",
    "AnimaTiledSamplerFromTiles": "Anima Tiled Sampler From Tiles",
    "AnimaTilePreview": "Anima Tile Previewer",
}
