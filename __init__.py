import server
from aiohttp import web
from .core.immich_api import immich_api
from .nodes.immich_nodes import NODE_CLASS_MAPPINGS as IMMICH_NCM, NODE_DISPLAY_NAME_MAPPINGS as IMMICH_NNM
from .nodes.image_nodes import NODE_CLASS_MAPPINGS as IMAGE_NCM, NODE_DISPLAY_NAME_MAPPINGS as IMAGE_NNM
from .nodes.save_nodes import NODE_CLASS_MAPPINGS as SAVE_NCM, NODE_DISPLAY_NAME_MAPPINGS as SAVE_NNM
from .nodes.anima_upscaler import NODE_CLASS_MAPPINGS as ANIMA_NCM, NODE_DISPLAY_NAME_MAPPINGS as ANIMA_NNM
from .nodes.video_nodes import NODE_CLASS_MAPPINGS as VIDEO_NCM, NODE_DISPLAY_NAME_MAPPINGS as VIDEO_NNM
from .core.kjnodes_patch import patch_kjnodes_preview_override

# Automatically patch KJNodes ModelPreviewOverride for AMD / CPU fast MP4 previews
try:
    patch_kjnodes_preview_override()
except Exception as e:
    print(f"FerrahNodes: Note: KJNodes patch skipped ({e})")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Merge node mappings
NODE_CLASS_MAPPINGS.update(IMMICH_NCM)
NODE_DISPLAY_NAME_MAPPINGS.update(IMMICH_NNM)

NODE_CLASS_MAPPINGS.update(IMAGE_NCM)
NODE_DISPLAY_NAME_MAPPINGS.update(IMAGE_NNM)

NODE_CLASS_MAPPINGS.update(SAVE_NCM)
NODE_DISPLAY_NAME_MAPPINGS.update(SAVE_NNM)

NODE_CLASS_MAPPINGS.update(ANIMA_NCM)
NODE_DISPLAY_NAME_MAPPINGS.update(ANIMA_NNM)

NODE_CLASS_MAPPINGS.update(VIDEO_NCM)
NODE_DISPLAY_NAME_MAPPINGS.update(VIDEO_NNM)

# API Routes
@server.PromptServer.instance.routes.get("/immich/get_albums")
async def get_immich_albums(request):
    try:
        albums = immich_api.get_albums()
        if "error" in albums:
            return web.json_response(albums, status=400)
            
        # Return simplified list for the frontend
        result = [{"id": a.get("id"), "name": a.get("albumName")} for a in albums]
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']