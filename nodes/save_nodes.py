import folder_paths, os, random 
from nodes import SaveImage

class SessionSaveImage(SaveImage):
    def __init__(self):
        super().__init__()
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = "_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 8

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "image", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "FCN/image"

NODE_CLASS_MAPPINGS = {
    "SessionIMGSave": SessionSaveImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SessionIMGSave": "Save Image per Session"
}
