import server
from aiohttp import web
from .core.immich_api import immich_api
from .nodes.immich_nodes import NODE_CLASS_MAPPINGS as IMMICH_NCM, NODE_DISPLAY_NAME_MAPPINGS as IMMICH_NNM
from .nodes.image_nodes import NODE_CLASS_MAPPINGS as IMAGE_NCM, NODE_DISPLAY_NAME_MAPPINGS as IMAGE_NNM
from .nodes.save_nodes import NODE_CLASS_MAPPINGS as SAVE_NCM, NODE_DISPLAY_NAME_MAPPINGS as SAVE_NNM

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Merge node mappings
NODE_CLASS_MAPPINGS.update(IMMICH_NCM)
NODE_DISPLAY_NAME_MAPPINGS.update(IMMICH_NNM)

NODE_CLASS_MAPPINGS.update(IMAGE_NCM)
NODE_DISPLAY_NAME_MAPPINGS.update(IMAGE_NNM)

NODE_CLASS_MAPPINGS.update(SAVE_NCM)
NODE_DISPLAY_NAME_MAPPINGS.update(SAVE_NNM)

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