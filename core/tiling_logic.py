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

from .tiled_utils import (
    pil_to_tensor, create_latent_blend_mask, crop_cond, 
    patch_anima_rope, sample_tile, get_vae_spatial_compression, 
    get_model_patch_spatial, get_pixel_alignment, crop_noise_for_tile
)



def calculate_tiling_grid(
    image, vae_spatial, patch_spatial, pixel_align,
    tiling_strategy, tile_size_mode, target_tile_size, min_tile_size,
    tile_width, tile_height, padding, mask_blur, adaptive_tiling,
    detail_percentile=0.85,
    target_w=None, target_h=None, scale_factor=None
):
    batch_size = image.shape[0]
    batch_configs = []

    def get_intervals(total_size, step_size, min_sz):
        intervals = []
        curr = 0
        while curr < total_size:
            nxt = min(curr + step_size, total_size)
            if nxt == total_size and (nxt - curr) < min_sz and len(intervals) > 0:
                intervals[-1] = (intervals[-1][0], total_size)
            else:
                intervals.append((curr, nxt))
            curr = nxt
        return intervals

    for b in range(batch_size):
        img_b = image[b:b+1]
        b_w, b_h = img_b.shape[2], img_b.shape[1]

        t_w = target_w if target_w is not None else b_w
        t_h = target_h if target_h is not None else b_h
        
        b_scale_factor = scale_factor if scale_factor is not None else (float(t_w) / float(b_w))

        # --- Laplacian Variance Calculation ---
        gray_lr = (img_b[..., 0] * 0.299 + img_b[..., 1] * 0.587 + img_b[..., 2] * 0.114).unsqueeze(1)
        blur_kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=img_b.device) / 9.0
        gray_lr = F.conv2d(gray_lr, blur_kernel.to(gray_lr.dtype), padding=1)

        lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=img_b.device).view(1, 1, 3, 3)
        lap_map_lr = F.conv2d(gray_lr, lap_kernel.to(gray_lr.dtype), padding=1).abs()

        # --- NOVO LIMIAR DINÂMICO (QUANTIL) ---
        # Define que apenas o top de detalhes absolutos da imagem (baseado no detail_percentile) será considerado
        # Dica: Se quiser expor isso no seu node do Comfy depois, chame de "Detail Focus" (0.0 a 1.0)
        
        # Transforma o tensor em float32 pra evitar erro de compatibilidade do quantile com float16
        flat_lap = lap_map_lr.to(torch.float32).view(-1)
        
        # Encontra o valor exato que separa o top 15% do resto
        dynamic_threshold = torch.quantile(flat_lap, detail_percentile).item()
        
        # Mantém um absolute_floor bem baixo apenas para evitar que imagens
        # completamente pretas/sólidas tentem subdividir micro-ruído invisível
        absolute_floor = 0.005
        final_threshold = max(dynamic_threshold, absolute_floor)
        
        # Zera tudo que está abaixo do limiar
        lap_map_lr = torch.where(lap_map_lr > final_threshold, lap_map_lr, torch.zeros_like(lap_map_lr))
        # --------------------------------------

        def get_region_variance(x1, y1, x2, y2):
            lr_x1 = int(x1 / b_scale_factor)
            lr_y1 = int(y1 / b_scale_factor)
            lr_x2 = min(int(x2 / b_scale_factor), lap_map_lr.shape[3])
            lr_y2 = min(int(y2 / b_scale_factor), lap_map_lr.shape[2])
            if lr_x2 <= lr_x1 or lr_y2 <= lr_y1:
                return 0.0
            tile_lap = lap_map_lr[:, :, lr_y1:lr_y2, lr_x1:lr_x2]
            return float(tile_lap.mean()) if tile_lap.numel() > 0 else 0.0

        # --- Grid Generation ---
        if tile_size_mode == "Adaptive (Quadtree)":
            baseline_variances = []
            grid_sz = max(pixel_align, min_tile_size)
            
            x_intervals = get_intervals(t_w, grid_sz, min_tile_size)
            y_intervals = get_intervals(t_h, grid_sz, min_tile_size)
            
            for gy1, gy2 in y_intervals:
                for gx1, gx2 in x_intervals:
                    baseline_variances.append(get_region_variance(gx1, gy1, gx2, gy2))
            
            baseline_variances = sorted(baseline_variances)
            if baseline_variances:
                med_idx = min(len(baseline_variances) - 1, max(0, int(len(baseline_variances) * 0.50)))
                split_threshold = max(0.002, baseline_variances[med_idx] * 0.8)
                roi_idx = min(len(baseline_variances) - 1, max(0, int(len(baseline_variances) * 0.80)))
                roi_threshold = max(0.008, baseline_variances[roi_idx])
                ref_var = baseline_variances[roi_idx]
            else:
                split_threshold, roi_threshold, ref_var = 0.003, 0.01, 1.0

            max_depth = 2 if min_tile_size < target_tile_size else 1
            base_block_size = target_tile_size * 2

            def build_tree(x1, y1, x2, y2, depth):
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    return None
                
                var = get_region_variance(x1, y1, x2, y2)
                
                if var <= 0.0001:
                    return None
                
                should_split = False
                
                if depth < max_depth and (w // 2 >= min_tile_size) and (h // 2 >= min_tile_size):
                    if depth == 0 and var > split_threshold:
                        should_split = True
                    elif depth == 1 and var > roi_threshold:
                        should_split = True
                            
                    if w >= target_tile_size * (2 ** (max_depth - depth - 1)) or h >= target_tile_size * (2 ** (max_depth - depth - 1)):
                        should_split = True
                
                node = {
                    "x1": x1, "y1": y1, "w": w, "h": h, "depth": depth, "var": var,
                    "is_leaf": not should_split, "children": []
                }
                
                if should_split:
                    cx = ((x1 + w // 2) // pixel_align) * pixel_align
                    cy = ((y1 + h // 2) // pixel_align) * pixel_align
                    
                    cx = max(x1 + pixel_align, min(cx, x2 - pixel_align))
                    cy = max(y1 + pixel_align, min(cy, y2 - pixel_align))
                        
                    children_nodes = [
                        build_tree(x1, y1, cx, cy, depth + 1),
                        build_tree(cx, y1, x2, cy, depth + 1),
                        build_tree(x1, cy, cx, y2, depth + 1),
                        build_tree(cx, cy, x2, y2, depth + 1)
                    ]
                    
                    node["children"] = [c for c in children_nodes if c is not None]
                    
                    if not node["children"]:
                        node["is_leaf"] = True
                        return node
                    
                return node

            base_blocks = []
            x_base_intervals = get_intervals(t_w, base_block_size, min_tile_size)
            y_base_intervals = get_intervals(t_h, base_block_size, min_tile_size)
            
            for gy1, gy2 in y_base_intervals:
                for gx1, gx2 in x_base_intervals:
                    block_node = build_tree(gx1, gy1, gx2, gy2, depth=0)
                    if block_node is not None:
                        base_blocks.append(block_node)

            if tiling_strategy == "Chess":
                base_blocks.sort(key=lambda c: ((c["x1"] // base_block_size) + (c["y1"] // base_block_size)) % 2)
            elif tiling_strategy == "Reverse Chess":
                base_blocks.sort(key=lambda c: (((c["x1"] // base_block_size) + (c["y1"] // base_block_size)) % 2) == 0)
            elif tiling_strategy == "Spiral":
                base_blocks.sort(key=lambda c: (c["x1"] + c["w"]/2 - t_w/2)**2 + (c["y1"] + c["h"]/2 - t_h/2)**2)
            elif tiling_strategy == "Detail-First":
                base_blocks.sort(key=lambda c: c["var"], reverse=True)

            def sort_node(node):
                if node["is_leaf"]: return
                if tiling_strategy == "Chess":
                    node["children"].sort(key=lambda c: ((c["x1"] + c["y1"]) // pixel_align) % 2)
                elif tiling_strategy == "Reverse Chess":
                    node["children"].sort(key=lambda c: (((c["x1"] + c["y1"]) // pixel_align) % 2) == 0)
                elif tiling_strategy == "Spiral":
                    node["children"].sort(key=lambda c: (c["x1"] + c["w"]/2 - t_w/2)**2 + (c["y1"] + c["h"]/2 - t_h/2)**2)
                elif tiling_strategy == "Detail-First":
                    node["children"].sort(key=lambda c: c["var"], reverse=True)
                for child in node["children"]: sort_node(child)
                    
            for block in base_blocks: sort_node(block)
            
        else:
            if tile_size_mode == "Auto":
                cols, rows = max(1, round(t_w / target_tile_size)), max(1, round(t_h / target_tile_size))
                current_tile_width = max(pixel_align * 4, math.ceil((t_w / cols) / pixel_align) * pixel_align)
                current_tile_height = max(pixel_align * 4, math.ceil((t_h / rows) / pixel_align) * pixel_align)
            else:
                current_tile_width, current_tile_height = (tile_width // pixel_align) * pixel_align, (tile_height // pixel_align) * pixel_align

            rows, cols = math.ceil(t_h / current_tile_height), math.ceil(t_w / current_tile_width)
            base_blocks = []
            
            for yi in range(rows):
                for xi in range(cols):
                    cx1, cy1 = xi * current_tile_width, yi * current_tile_height
                    cx2 = min(cx1 + current_tile_width, t_w)
                    cy2 = min(cy1 + current_tile_height, t_h)
                    var = get_region_variance(cx1, cy1, cx2, cy2)
                    
                    if var > 0.0001: 
                        base_blocks.append({
                            "x1": cx1, "y1": cy1, 
                            "w": cx2 - cx1, 
                            "h": cy2 - cy1, 
                            "depth": 0, 
                            "var": var, 
                            "is_leaf": True, "children": []
                        })

            if tiling_strategy == "Chess": 
                base_blocks.sort(key=lambda c: ((c["x1"] // current_tile_width) + (c["y1"] // current_tile_height)) % 2)
            elif tiling_strategy == "Reverse Chess": 
                base_blocks.sort(key=lambda c: (((c["x1"] // current_tile_width) + (c["y1"] // current_tile_height)) % 2) == 0)
            elif tiling_strategy == "Spiral": 
                base_blocks.sort(key=lambda c: (c["x1"] + c["w"]/2 - t_w/2)**2 + (c["y1"] + c["h"]/2 - t_h/2)**2)
            elif tiling_strategy == "Detail-First": 
                base_blocks.sort(key=lambda c: c["var"], reverse=True)
            ref_var = max(0.005, np.median([b["var"] for b in base_blocks])) if base_blocks else 1.0

        batch_configs.append({
            "base_blocks": base_blocks,
            "ref_var": ref_var,
            "target_w": t_w,
            "target_h": t_h
        })

    # Temporary debug print for coordinates and metadata (without latent tensors/arrays)
    print(f"--- [ANIMA DEBUG] TILING GRID CONFIGS (Batch Size: {len(batch_configs)}) ---")
    for idx_b, config in enumerate(batch_configs):
        print(f"Batch Item {idx_b}: target_w={config['target_w']}, target_h={config['target_h']}, ref_var={config['ref_var']:.6f}")
        
        def get_leaves_local(node):
            if node["is_leaf"]:
                yield node
            else:
                for child in node["children"]:
                    yield from get_leaves_local(child)
        
        flat_tiles = []
        for block in config["base_blocks"]:
            flat_tiles.extend(list(get_leaves_local(block)))
            
        print(f"Total tiles: {len(flat_tiles)}")
        for idx_t, tile in enumerate(flat_tiles):
            x1, y1, w, h = tile["x1"], tile["y1"], tile["w"], tile["h"]
            var = tile.get("var", 0.0)
            depth = tile.get("depth", 0)
            
            # Replicate the padding calculation
            sample_w = max(w + padding * 2, target_tile_size)
            sample_h = max(h + padding * 2, target_tile_size)
            
            cx = x1 + w // 2
            cy = y1 + h // 2
            
            px1 = (cx - sample_w // 2) // pixel_align * pixel_align
            py1 = (cy - sample_h // 2) // pixel_align * pixel_align
            px2 = px1 + sample_w
            py2 = py1 + sample_h
            
            px1 = max(0, px1)
            py1 = max(0, py1)
            px2 = min(config["target_w"], px2)
            py2 = min(config["target_h"], py2)
            
            px1 = (px1 // pixel_align) * pixel_align
            py1 = (py1 // pixel_align) * pixel_align
            px2 = (px2 // pixel_align) * pixel_align
            py2 = (py2 // pixel_align) * pixel_align
            
            pw = px2 - px1
            ph = py2 - py1
            
            # skipped status
            is_skipped = False
            if adaptive_tiling and tile_size_mode == "Adaptive (Quadtree)":
                ref_var = config["ref_var"]
                var_ratio = var / ref_var if (ref_var is not None and ref_var > 1e-8) else 1.0
                if var < 0.002 or var_ratio < 0.15:
                    is_skipped = True
                    
            print(f"  Tile {idx_t:02d}: x1={x1:4d}, y1={y1:4d}, w={w:4d}, h={h:4d} | "
                  f"padding: px1={px1:4d}, py1={py1:4d}, pw={pw:4d}, ph={ph:4d} | "
                  f"var={var:.6f}, depth={depth}, skipped={is_skipped}")
    print("--- [ANIMA DEBUG] END ---")

    return batch_configs




def draw_tiling_preview_image(
    image: torch.Tensor,
    batch_configs: list,
    tiling_strategy: str,
    adaptive_tiling: bool,
    tile_size_mode: str,
    padding: int = 0,
    target_tile_size: int = 1024,
    pixel_align: int = 16,
    draw_tiles: bool = True,
    draw_padding: bool = False
) -> torch.Tensor:
    """
    Generate preview images with overlays representing the tiling layout.
    """
    def get_leaves(node):
        if node["is_leaf"]:
            yield node
        else:
            for child in node["children"]:
                yield from get_leaves(child)

    batch_size = image.shape[0]
    output_images = []

    for b in range(batch_size):
        img_b = image[b]  # (H, W, C)
        img_np = (img_b.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # Determine base font size scaled to image size
        base_font_size = max(14, int(min(img_pil.width, img_pil.height) * 0.015))
        border_width = max(2, int(base_font_size * 0.1))
        padding_border_width = max(1, border_width // 2)

        # Attempt to load a nice font, fall back to default
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=base_font_size)
        except IOError:
            try:
                font = ImageFont.truetype("LiberationSans-Regular.ttf", size=base_font_size)
            except IOError:
                try:
                    font = ImageFont.load_default(size=base_font_size)
                except TypeError:
                    font = ImageFont.load_default()
        
        # We need an RGBA overlay for translucent fills
        overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, 'RGBA')
        
        config = batch_configs[b]
        base_blocks = config["base_blocks"]
        ref_var = config["ref_var"]
        
        # Flatten leaf tiles in processing order
        leaf_tiles = []
        for block in base_blocks:
            leaf_tiles.extend(list(get_leaves(block)))
            
        # Phase 1: Draw Core Tiles
        if draw_tiles:
            for idx, node in enumerate(leaf_tiles):
                x1, y1, w, h = node["x1"], node["y1"], node["w"], node["h"]
                var = node["var"]
                
                # Check if skipped
                is_skipped = False
                if adaptive_tiling and tile_size_mode == "Adaptive (Quadtree)":
                    var_ratio = var / ref_var if (ref_var is not None and ref_var > 1e-8) else 1.0
                    if var < 0.002 or var_ratio < 0.15:
                        is_skipped = True
                
                x2, y2 = x1 + w, y1 + h
                
                if is_skipped:
                    # Translucent red/gray overlay
                    draw.rectangle([x1, y1, x2, y2], fill=(128, 128, 128, 100), outline=(220, 50, 50, 180), width=border_width)
                    # Dotted line approximation for skipped tiles
                    dash_len = max(4, int(base_font_size * 0.4))
                    for step in range(x1, x2, dash_len * 2):
                        draw.line([step, y1, min(step + dash_len, x2), y1], fill=(220, 50, 50, 255), width=border_width)
                        draw.line([step, y2, min(step + dash_len, x2), y2], fill=(220, 50, 50, 255), width=border_width)
                    for step in range(y1, y2, dash_len * 2):
                        draw.line([x1, step, x1, min(step + dash_len, y2)], fill=(220, 50, 50, 255), width=border_width)
                        draw.line([x2, step, x2, min(step + dash_len, y2)], fill=(220, 50, 50, 255), width=border_width)
                else:
                    # Translucent cool blue/cyan overlay (15% opacity fill, 80% opacity border)
                    draw.rectangle([x1, y1, x2, y2], fill=(0, 150, 255, 38), outline=(0, 150, 255, 204), width=border_width)

        # Phase 2: Draw Padded Overlaps as orange blocks on top of blue tiles
        if draw_padding and padding > 0:
            for idx, node in enumerate(leaf_tiles):
                x1, y1, w, h = node["x1"], node["y1"], node["w"], node["h"]
                
                sample_w = max(w + padding * 2, target_tile_size)
                sample_h = max(h + padding * 2, target_tile_size)
                
                cx = x1 + w // 2
                cy = y1 + h // 2
                
                px1 = (cx - sample_w // 2) // pixel_align * pixel_align
                py1 = (cy - sample_h // 2) // pixel_align * pixel_align
                px2 = px1 + sample_w
                py2 = py1 + sample_h
                
                px1 = max(0, px1)
                py1 = max(0, py1)
                px2 = min(img_pil.width, px2)
                py2 = min(img_pil.height, py2)
                
                px1 = (px1 // pixel_align) * pixel_align
                py1 = (py1 // pixel_align) * pixel_align
                px2 = (px2 // pixel_align) * pixel_align
                py2 = (py2 // pixel_align) * pixel_align
                
                # Draw padding block region (translucent orange fill and outline)
                draw.rectangle([px1, py1, px2, py2], fill=(255, 110, 0, 25), outline=(255, 110, 0, 180), width=padding_border_width)

        # Phase 3: Draw Badge text labels on top of all layers
        if draw_tiles:
            for idx, node in enumerate(leaf_tiles):
                x1, y1, w, h = node["x1"], node["y1"], node["w"], node["h"]
                var = node["var"]
                
                # Check if skipped
                is_skipped = False
                if adaptive_tiling and tile_size_mode == "Adaptive (Quadtree)":
                    var_ratio = var / ref_var if (ref_var is not None and ref_var > 1e-8) else 1.0
                    if var < 0.002 or var_ratio < 0.15:
                        is_skipped = True
                
                x2, y2 = x1 + w, y1 + h
                
                badge_text = f"#{idx+1}"
                if is_skipped:
                    badge_text += " [SKIP]"
                else:
                    badge_text += f" (v:{var:.4f})"
                    
                # Measure text size
                try:
                    text_bbox = draw.textbbox((0, 0), badge_text, font=font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                except AttributeError:
                    # Fallback for older PIL versions
                    text_w, text_h = draw.textsize(badge_text, font=font)
                    
                badge_w = text_w + int(base_font_size * 0.6)
                badge_h = text_h + int(base_font_size * 0.4)
                
                bx1 = x1 + border_width + 4
                by1 = y1 + border_width + 4
                bx2 = bx1 + badge_w
                by2 = by1 + badge_h
                
                # Clamp badge within the tile
                if bx2 > x2:
                    bx2 = x2 - border_width - 4
                    bx1 = max(x1, bx2 - badge_w)
                if by2 > y2:
                    by2 = y2 - border_width - 4
                    by1 = max(y1, by2 - badge_h)
                    
                badge_fill = (0, 0, 0, 180) if not is_skipped else (50, 0, 0, 200)
                draw.rectangle([bx1, by1, bx2, by2], fill=badge_fill, outline=(255, 255, 255, 100), width=1)
                
                text_fill = (255, 255, 255, 255) if not is_skipped else (255, 150, 150, 255)
                draw.text((bx1 + int(base_font_size * 0.3), by1 + int(base_font_size * 0.15)), badge_text, fill=text_fill, font=font)

            
        # Composite overlay
        img_composited = Image.alpha_composite(img_pil.convert('RGBA'), overlay)
        img_rgb = img_composited.convert('RGB')
        img_tensor = pil_to_tensor(img_rgb)  # Shape (1, H, W, C)
        output_images.append(img_tensor)

    return torch.cat(output_images, dim=0)


def execute_tiled_sampling_loop(
    model, positive, negative, latent_b_batch, image,
    vae_spatial, patch_spatial, pixel_align,
    seed, steps, cfg, sampler_name, scheduler, denoise,
    batch_configs, padding, mask_blur, adaptive_tiling,
    target_tile_size, tile_size_mode
):
    batch_size = latent_b_batch.shape[0]
    final_batch_outputs = []

    def get_leaves(node):
        if node["is_leaf"]:
            yield node
        else:
            for child in node["children"]:
                yield from get_leaves(child)

    # Count expected sampler steps for this batch
    total_expected_steps = 0
    for b in range(batch_size):
        config = batch_configs[b]
        base_blocks = config["base_blocks"]
        ref_var = config["ref_var"]
        
        def count_expected_steps(node):
            if node["is_leaf"]:
                actual_denoise = denoise
                if adaptive_tiling and tile_size_mode == "Adaptive (Quadtree)":
                    var_ratio, abs_var = node["var"] / ref_var, node["var"] if (ref_var is not None and ref_var > 1e-8) else 1.0
                    if abs_var < 0.002 or var_ratio < 0.15:
                        return max(1, int(steps * denoise))
                    if var_ratio < 0.5:
                        actual_denoise = denoise * (0.5 + 0.5 * var_ratio)
                return max(1, int(steps * actual_denoise))
            else:
                return sum(count_expected_steps(c) for c in node["children"])

        batch_steps = sum(count_expected_steps(b) for b in base_blocks)
        total_expected_steps += batch_steps

    # Instantiate global ProgressBar
    pbar = comfy.utils.ProgressBar(total_expected_steps)
    completed_steps = 0

    # ---------------------------------------------------------------------------
    # Main Batch Processing Loop
    # ---------------------------------------------------------------------------
    pos_embedder_to_restore = None
    orig_pos_embedder_forward = None

    try:
        for b in range(batch_size):
            print(f"[ANIMA] Processing batch element {b+1}/{batch_size}")
            
            config = batch_configs[b]
            base_blocks = config["base_blocks"]
            target_w = config["target_w"]
            target_h = config["target_h"]
            ref_var = config["ref_var"]
            total_base_blocks = len(base_blocks)
            
            img_b = image[b:b+1]  # (1, H, W, C)
            b_w, b_h = img_b.shape[2], img_b.shape[1]

            canvas_w, canvas_h = target_w, target_h
            print(f"[ANIMA] Target Canvas: {canvas_w}x{canvas_h}")

            raw_latent_2x = latent_b_batch[b:b+1].clone()
            latent_b = raw_latent_2x.clone()

            noise_divisor = vae_spatial
            global_noise_2x = comfy.sample.prepare_noise(raw_latent_2x, seed, None)

            # Recursive processing of tile nodes in latent space
            def process_tile_node(node, parent_node=None, child_idx=0, base_idx=0, base_block_state=None):
                nonlocal latent_b, pos_embedder_to_restore, orig_pos_embedder_forward, completed_steps
                
                x1, y1, actual_tw, actual_th, depth = node["x1"], node["y1"], node["w"], node["h"], node["depth"]
                
                if parent_node is None:
                    leaves_count = sum(1 for leaf in get_leaves(node))
                    base_block_state = {"sub_idx": 0, "leaves_count": leaves_count}
                    if leaves_count > 1:
                        print(f"[ANIMA] 📦 Processing Base Tile {base_idx+1}/{total_base_blocks} | Core: ({x1},{y1}) {actual_tw}x{actual_th} | Split into {leaves_count} sub-tiles")
                
                if node["is_leaf"]:
                    actual_denoise = denoise
                    if adaptive_tiling and tile_size_mode == "Adaptive (Quadtree)":
                        var_ratio, abs_var = node["var"] / ref_var, node["var"] if (ref_var is not None and ref_var > 1e-8) else 1.0
                        if abs_var < 0.002 or var_ratio < 0.15:
                            # Skip tile completely
                            tile_steps = max(1, int(steps * denoise))
                            completed_steps += tile_steps
                            pbar.update_absolute(completed_steps)
                            
                            sub_idx = base_block_state["sub_idx"]
                            leaves_count = base_block_state["leaves_count"]
                            base_block_state["sub_idx"] += 1
                            prefix = "└──" if sub_idx == leaves_count - 1 else "├──"
                            if leaves_count > 1:
                                print(f"[ANIMA]   {prefix} Sub-tile {sub_idx+1}/{leaves_count} | Skipped (low variance: {abs_var:.6f}, ratio: {var_ratio:.4f})")
                            else:
                                print(f"[ANIMA] 📦 Processing Base Tile {base_idx+1}/{total_base_blocks} (No Split) | Skipped (low variance: {abs_var:.6f}, ratio: {var_ratio:.4f})")
                            return
                        
                        if var_ratio < 0.5:
                            actual_denoise = denoise * (0.5 + 0.5 * var_ratio)

                    # Determine the coordinates in latent space
                    lx1_dst = x1 // noise_divisor
                    ly1_dst = y1 // noise_divisor
                    lx2_dst = (x1 + actual_tw) // noise_divisor
                    ly2_dst = (y1 + actual_th) // noise_divisor
                    tile_w_lat = lx2_dst - lx1_dst
                    tile_h_lat = ly2_dst - ly1_dst
                    
                    # Apply context padding to define crop region in 2x canvas
                    sample_w_2x = max(actual_tw + padding * 2, target_tile_size)
                    sample_h_2x = max(actual_th + padding * 2, target_tile_size)
                    
                    cx_2x = x1 + actual_tw // 2
                    cy_2x = y1 + actual_th // 2
                    
                    c_x1_2x = (cx_2x - sample_w_2x // 2) // pixel_align * pixel_align
                    c_y1_2x = (cy_2x - sample_h_2x // 2) // pixel_align * pixel_align
                    c_x2_2x = c_x1_2x + sample_w_2x
                    c_y2_2x = c_y1_2x + sample_h_2x
                    
                    # Clamp crop boundaries to target canvas size
                    c_x1_2x = max(0, c_x1_2x)
                    c_y1_2x = max(0, c_y1_2x)
                    c_x2_2x = min(canvas_w, c_x2_2x)
                    c_y2_2x = min(canvas_h, c_y2_2x)
                    
                    # Re-align clamped crop region to pixel_align
                    c_x1_2x = (c_x1_2x // pixel_align) * pixel_align
                    c_y1_2x = (c_y1_2x // pixel_align) * pixel_align
                    c_x2_2x = (c_x2_2x // pixel_align) * pixel_align
                    c_y2_2x = (c_y2_2x // pixel_align) * pixel_align
                    
                    crop_region_2x = (c_x1_2x, c_y1_2x, c_x2_2x, c_y2_2x)
                    real_w_2x = c_x2_2x - c_x1_2x
                    real_h_2x = c_y2_2x - c_y1_2x

                    # Exclusive Latent Slicing: Crop directly from raw_latent_2x instead of VAE encoding pixels
                    if raw_latent_2x.ndim == 5:
                        cropped_latent = raw_latent_2x[:, :, :, crop_region_2x[1] // noise_divisor:crop_region_2x[3] // noise_divisor, crop_region_2x[0] // noise_divisor:crop_region_2x[2] // noise_divisor].clone()
                    else:
                        cropped_latent = raw_latent_2x[:, :, crop_region_2x[1] // noise_divisor:crop_region_2x[3] // noise_divisor, crop_region_2x[0] // noise_divisor:crop_region_2x[2] // noise_divisor].clone()

                    if tile_h_lat <= 0 or tile_w_lat <= 0:
                        tile_steps = max(1, int(steps * actual_denoise))
                        completed_steps += tile_steps
                        pbar.update_absolute(completed_steps)
                        return

                    # Slice global noise using the utility function
                    tile_noise = crop_noise_for_tile(
                        global_noise_2x, crop_region_2x, noise_divisor,
                        target_size=(cropped_latent.shape[-1], cropped_latent.shape[-2])
                    )

                    # Print structured hierarchical log
                    if tile_size_mode == "Adaptive (Quadtree)":
                        sub_idx = base_block_state["sub_idx"]
                        leaves_count = base_block_state["leaves_count"]
                        base_block_state["sub_idx"] += 1
                        prefix = "└──" if sub_idx == leaves_count - 1 else "├──"
                        if leaves_count > 1:
                            print(f"[ANIMA]   {prefix} Sub-tile {sub_idx+1}/{leaves_count} | Core: ({x1},{y1}) {actual_tw}x{actual_th} -> Padding: {real_w_2x}x{real_h_2x} | Denoise: {actual_denoise:.4f}")
                        else:
                            print(f"[ANIMA] 📦 Processing Base Tile {base_idx+1}/{total_base_blocks} (No Split) | Core: ({x1},{y1}) {actual_tw}x{actual_th} -> Padding: {real_w_2x}x{real_h_2x} | Denoise: {actual_denoise:.4f}")
                    else:
                        print(f"[ANIMA] 📦 Processing Tile {base_idx+1}/{total_base_blocks} | Core: ({x1},{y1}) {actual_tw}x{actual_th} -> Padding: {real_w_2x}x{real_h_2x} | Denoise: {actual_denoise:.4f}")

                    lx1_src = max(0, x1 // noise_divisor - crop_region_2x[0] // noise_divisor)
                    ly1_src = max(0, y1 // noise_divisor - crop_region_2x[1] // noise_divisor)

                    # Create latent blend mask for composition and noise masking
                    lblend = max(1, mask_blur // noise_divisor) if mask_blur > 0 else 0
                    l_core_x1 = lx1_src
                    l_core_y1 = ly1_src
                    l_core_x2 = lx1_src + tile_w_lat
                    l_core_y2 = ly1_src + tile_h_lat

                    latent_blend_mask = create_latent_blend_mask(
                        cropped_latent.shape[-2], cropped_latent.shape[-1], 
                        l_core_x1, l_core_y1, l_core_x2, l_core_y2, 
                        lblend, device=cropped_latent.device
                    )

                    cropped_pos = crop_cond(positive, crop_region_2x, (b_w, b_h), (canvas_w, canvas_h), (real_w_2x, real_h_2x), divisor=noise_divisor)
                    cropped_neg = crop_cond(negative, crop_region_2x, (b_w, b_h), (canvas_w, canvas_h), (real_w_2x, real_h_2x), divisor=noise_divisor)

                    latent = {
                        "samples": cropped_latent,
                        "noise_mask": latent_blend_mask.unsqueeze(0)
                    }

                    try: dm = model.model.diffusion_model
                    except AttributeError: dm = None

                    if dm is not None and hasattr(dm, "pos_embedder"):
                        pos_embedder_to_restore = dm.pos_embedder
                        shift_x, shift_y = (crop_region_2x[0] // noise_divisor) // patch_spatial, (crop_region_2x[1] // noise_divisor) // patch_spatial
                        orig_pos_embedder_forward = patch_anima_rope(dm.pos_embedder, shift_x, shift_y)
                    else:
                        orig_pos_embedder_forward = None

                    # Progress indicators
                    completed_steps_at_start = completed_steps
                    last_step_seen = -1

                    def sampler_callback(step, x0, x, total_steps):
                        nonlocal completed_steps, last_step_seen
                        if step != last_step_seen:
                            if last_step_seen != -1:
                                steps_diff = step - last_step_seen
                                if steps_diff > 0:
                                    completed_steps += steps_diff
                            last_step_seen = step
                            pbar.update_absolute(completed_steps)
                        
                        current_step = step + 1
                        if total_steps > 0:
                            pct = int(current_step / total_steps * 100)
                            bar_length = 20
                            filled_length = int(bar_length * current_step // total_steps)
                            bar = "█" * filled_length + "░" * (bar_length - filled_length)
                            print(f"\r[ANIMA]   └── Sampling Steps: [{bar}] {pct}% ({current_step}/{total_steps})", end="", flush=True)

                    try:
                        # Pass sliced noise and step callback directly to sample_tile
                        sampled = sample_tile(
                            model, cropped_pos, cropped_neg, sampler_name, scheduler, steps, cfg, latent, seed, actual_denoise,
                            callback=sampler_callback, disable_pbar=True, disable_noise=False
                        )
                    finally:
                        if orig_pos_embedder_forward is not None:
                            pos_embedder_to_restore.generate_embeddings = orig_pos_embedder_forward
                            pos_embedder_to_restore = None
                            orig_pos_embedder_forward = None
                        
                        # Clean up step progress bar for this tile
                        tile_steps = last_step_seen + 1 if last_step_seen != -1 else 0
                        expected_steps = int(steps * actual_denoise)
                        actual_steps_run = max(tile_steps, expected_steps)
                        
                        remaining = actual_steps_run - (completed_steps - completed_steps_at_start)
                        if remaining > 0:
                            completed_steps += remaining
                            pbar.update_absolute(completed_steps)
                        
                        if actual_steps_run > 0:
                            bar = "█" * 20
                            print(f"\r[ANIMA]   └── Sampling Steps: [{bar}] 100% ({actual_steps_run}/{actual_steps_run})", flush=True)

                    sampled_samples = sampled["samples"]

                    # Defensive check: ensure sampled_samples matches expected dimensions
                    s_h, s_w = sampled_samples.shape[-2], sampled_samples.shape[-1]
                    if s_h != cropped_latent.shape[-2] or s_w != cropped_latent.shape[-1]:
                        print(f"[ANIMA] Warning: latent shape mismatch. Sampled: {s_h}x{s_w}, Expected: {cropped_latent.shape[-2]}x{cropped_latent.shape[-1]}. Resizing...")
                        has_5d = (sampled_samples.ndim == 5)
                        if has_5d:
                            B, C, T, H, W = sampled_samples.shape
                            sampled_samples = sampled_samples.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                        
                        sampled_samples = F.interpolate(sampled_samples.float(), size=cropped_latent.shape[-2:], mode="nearest-exact").to(sampled_samples.dtype)
                        
                        if has_5d:
                            sampled_samples = sampled_samples.reshape(B, T, C, cropped_latent.shape[-2], cropped_latent.shape[-1]).permute(0, 2, 1, 3, 4)

                    # Paste processed tile back to canvas using blending mask
                    if latent_b.ndim == 5:
                        latent_blend_mask = latent_blend_mask.view(1, 1, 1, sampled_samples.shape[-2], sampled_samples.shape[-1])
                        latent_b[:, :, :, ly1_dst:ly2_dst, lx1_dst:lx2_dst] = (
                            sampled_samples[:, :, :, l_core_y1:l_core_y2, l_core_x1:l_core_x2] * latent_blend_mask[:, :, :, l_core_y1:l_core_y2, l_core_x1:l_core_x2] + 
                            latent_b[:, :, :, ly1_dst:ly2_dst, lx1_dst:lx2_dst] * (1.0 - latent_blend_mask[:, :, :, l_core_y1:l_core_y2, l_core_x1:l_core_x2])
                        )
                    else:
                        latent_blend_mask = latent_blend_mask.view(1, 1, sampled_samples.shape[-2], sampled_samples.shape[-1])
                        latent_b[:, :, ly1_dst:ly2_dst, lx1_dst:lx2_dst] = (
                            sampled_samples[:, :, l_core_y1:l_core_y2, l_core_x1:l_core_x2] * latent_blend_mask[:, :, l_core_y1:l_core_y2, l_core_x1:l_core_x2] + 
                            latent_b[:, :, ly1_dst:ly2_dst, lx1_dst:lx2_dst] * (1.0 - latent_blend_mask[:, :, l_core_y1:l_core_y2, l_core_x1:l_core_x2])
                        )

                    comfy.model_management.throw_exception_if_processing_interrupted()
                else:
                    for idx, child in enumerate(node["children"]):
                        process_tile_node(child, parent_node=node, child_idx=idx, base_idx=base_idx, base_block_state=base_block_state)

            # Process all base blocks recursively
            for b_idx, base_block in enumerate(base_blocks):
                process_tile_node(base_block, parent_node=None, child_idx=0, base_idx=b_idx)

            final_batch_outputs.append(latent_b)

            del raw_latent_2x, global_noise_2x, latent_b
            gc.collect()
            comfy.model_management.soft_empty_cache()

    finally:
        if pos_embedder_to_restore is not None and orig_pos_embedder_forward is not None:
            try: pos_embedder_to_restore.generate_embeddings = orig_pos_embedder_forward
            except Exception: pass

    return {"samples": torch.cat(final_batch_outputs, dim=0)}


def execute_tiled_sampling(
    model, positive, negative, latent_b_batch, image,
    vae_spatial, patch_spatial, pixel_align,
    seed, steps, cfg, sampler_name, scheduler, denoise,
    tiling_strategy, tile_size_mode, target_tile_size, min_tile_size,
    tile_width, tile_height, padding, mask_blur, adaptive_tiling,
    detail_percentile=0.85,
    scale_factor=None
):
    # 1. Calculate the grid configs for the batch
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
        target_w=latent_b_batch.shape[-1] * vae_spatial,
        target_h=latent_b_batch.shape[-2] * vae_spatial,
        scale_factor=scale_factor
    )

    # 2. Execute the sampling loop using the grid
    return execute_tiled_sampling_loop(
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
        batch_configs=batch_configs,
        padding=padding,
        mask_blur=mask_blur,
        adaptive_tiling=adaptive_tiling,
        target_tile_size=target_tile_size,
        tile_size_mode=tile_size_mode
    )