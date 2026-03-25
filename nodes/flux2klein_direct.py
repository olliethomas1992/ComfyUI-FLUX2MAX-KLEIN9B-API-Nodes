import io
import base64
import numpy as np
from PIL import Image
from .base import BaseFlux


def image_to_base64(image_tensor):
    """Convert a ComfyUI IMAGE tensor to base64 JPEG string."""
    img_array = (image_tensor[0].numpy() * 255).astype(np.uint8)
    pil_image = Image.fromarray(img_array).convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class Flux2Klein9bDirect(BaseFlux):
    """Flux 2 Klein 9B with direct IMAGE inputs — no separate base64 converter needed.
    Supports up to 4 reference images, matching the BFL API."""

    CATEGORY = "BFL/Flux2"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 5}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "width": ("INT", {"default": 0, "min": 0}),
                "height": ("INT", {"default": 0, "min": 0}),
                "seed": ("INT", {"default": -1}),
                "webhook_url": ("STRING", {"default": ""}),
                "webhook_secret": ("STRING", {"default": ""}),
                "config": ("BFL_CONFIG",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"

    def generate_image(
        self,
        prompt,
        safety_tolerance,
        output_format,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        width=0,
        height=0,
        seed=-1,
        webhook_url="",
        webhook_secret="",
        config=None,
    ):
        arguments = {
            "prompt": prompt,
            "safety_tolerance": safety_tolerance,
            "output_format": output_format,
        }

        image_slots = [
            ("input_image", image_1),
            ("input_image_2", image_2),
            ("input_image_3", image_3),
            ("input_image_4", image_4),
        ]
        for key, img in image_slots:
            if img is not None:
                arguments[key] = image_to_base64(img)

        if width > 0:
            arguments["width"] = width
        if height > 0:
            arguments["height"] = height
        if seed != -1:
            arguments["seed"] = seed
        if webhook_url:
            arguments["webhook_url"] = webhook_url
        if webhook_secret:
            arguments["webhook_secret"] = webhook_secret

        try:
            task_id = self.post_request("flux-2-klein-9b", arguments, config)
            if task_id:
                print(f"[BFL Flux2Klein9bDirect] Task ID '{task_id}'")
                return self.get_result(
                    task_id, output_format=output_format, config_override=config
                )
            return self.create_blank_image()
        except Exception as e:
            print(f"[BFL Flux2Klein9bDirect] Error: {str(e)}")
            return self.create_blank_image()


NODE_CLASS_MAPPINGS = {
    "Flux2Klein9bDirect_BFL": Flux2Klein9bDirect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2Klein9bDirect_BFL": "Flux 2 Klein 9B Direct (BFL)",
}
