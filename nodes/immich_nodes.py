import numpy as np
import io
from PIL import Image
import json
from datetime import datetime
import os
import folder_paths
from ..core.immich_api import immich_api
from ..core.utils import is_true, get_metadata_exif

# Check for AVIF support
try:
    import pillow_avif
    avif_supported = True
except ImportError:
    avif_supported = False

# Check for JXL support
try:
    import pillow_jxl
    jxl_supported = True
except ImportError:
    jxl_supported = False

class ImmichUpload:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "images": ("IMAGE",),
                "format": (["AVIF", "JXL"],),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "quality": ("INT", {"default": 80, "min": 1, "max": 100}),
                "save_locally": ("BOOLEAN", {"default": False}),
                "add_to_album": ("BOOLEAN", {"default": False}),
                "album_id": ("STRING", {"default": ""}),
                "embed_metadata": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "status",)
    FUNCTION = "upload"
    OUTPUT_NODE = True
    CATEGORY = "FCN/immich"

    def upload(self, enabled, images, format, filename_prefix, quality, save_locally, add_to_album, album_id, embed_metadata, prompt=None, extra_pnginfo=None, unique_id=None):
        enabled_bool = is_true(enabled)
        add_to_album_bool = is_true(add_to_album)
        save_locally_bool = is_true(save_locally)
        embed_metadata_bool = is_true(embed_metadata)

        if not enabled_bool:
            return (images, json.dumps({"status": "skipped"}),)

        if format == "AVIF" and not avif_supported:
            raise ImportError("Format selected is AVIF, but 'pillow-avif-plugin' is not installed.")
        if format == "JXL" and not jxl_supported:
            raise ImportError("Format selected is JXL, but 'pillow-jxl-plugin' is not installed.")

        # Extração inteligente do ID do álbum (formato: "Nome (ID)")
        if album_id and "(" in album_id and album_id.endswith(")"):
            extracted_id = album_id.split("(")[-1].rstrip(")")
            if len(extracted_id) > 10: # Validação simples de UUID
                album_id = extracted_id

        results = {"uploaded": [], "local": [], "errors": []}

        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            now = datetime.now().astimezone()

            kwargs = {"quality": quality}
            kwargs["exif"] = get_metadata_exif(img, prompt, extra_pnginfo, unique_id, embed_metadata_bool, now)

            if format == "AVIF":
                save_format = "AVIF"
                ext = "avif"
                mime_type = "image/avif"
                if quality == 100: kwargs["lossless"] = True
            elif format == "JXL":
                save_format = "JXL"
                ext = "jxl"
                mime_type = "image/jxl"
                if quality == 100: kwargs["lossless"] = True
            else:
                raise ValueError(f"Unsupported format: {format}")

            buffer = io.BytesIO()
            img.save(buffer, save_format, **kwargs)
            buffer.seek(0)

            file_basename = f"{filename_prefix}_{now.strftime('%Y%m%d_%H%M%S%f')}.{ext}"
            
            # Save locally if requested
            if save_locally_bool:
                try:
                    output_dir = folder_paths.get_output_directory()
                    full_output_path = os.path.join(output_dir, file_basename)
                    img.save(full_output_path, save_format, **kwargs)
                    results["local"].append(full_output_path)
                except Exception as e:
                    results["errors"].append(f"Local save error: {str(e)}")

            # Upload to Immich
            upload_result = immich_api.upload_asset(file_basename, buffer, mime_type, now)
            
            if "error" in upload_result:
                results["errors"].append(f"Upload error: {upload_result['error']}")
                # Fallback save if upload failed and not already saved
                if not save_locally_bool:
                    try:
                        output_dir = folder_paths.get_output_directory()
                        full_output_path = os.path.join(output_dir, file_basename)
                        img.save(full_output_path, save_format, **kwargs)
                        results["local"].append(full_output_path)
                    except Exception as e:
                        results["errors"].append(f"Fallback save error: {str(e)}")
            else:
                asset_id = upload_result.get('id')
                results["uploaded"].append(asset_id)
                
                if add_to_album_bool and album_id and asset_id:
                    album_result = immich_api.add_to_album(album_id, asset_id)
                    if "error" in album_result:
                        results["errors"].append(f"Album add error: {album_result['error']}")

        return (images, json.dumps(results),)

NODE_CLASS_MAPPINGS = {
    "immich_upload": ImmichUpload
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "immich_upload": "Immich Upload (AVIF/JXL)"
}
