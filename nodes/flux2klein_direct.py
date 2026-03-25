import io
import base64
import time
import requests
import numpy as np
from PIL import Image
import comfy.utils
from .base import BaseFlux, REQUEST_TIMEOUT
from .status import Status
from .config_node import get_config_loader


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
                return self._poll_with_progress(
                    task_id, output_format=output_format, config_override=config
                )
            return self.create_blank_image()
        except Exception as e:
            print(f"[BFL Flux2Klein9bDirect] Error: {str(e)}")
            return self.create_blank_image()

    def _poll_with_progress(self, task_id, output_format="jpeg", max_attempts=40, config_override=None):
        config_loader_instance = get_config_loader(config_override)
        headers = {"x-key": config_loader_instance.get_x_key()}
        get_url = config_loader_instance.create_url(f"get_result?id={task_id}")

        pbar = comfy.utils.ProgressBar(max_attempts)
        attempt = 1
        start_time = time.time()

        while attempt <= max_attempts:
            elapsed = time.time() - start_time
            try:
                print(f"[BFL Flux2Klein9bDirect] Poll {attempt}/{max_attempts} | {elapsed:.1f}s")
                result_response = requests.get(get_url, headers=headers, timeout=REQUEST_TIMEOUT)

                if result_response.status_code != 200:
                    print(f"[BFL Flux2Klein9bDirect] HTTP {result_response.status_code}")
                    pbar.update(1)
                    attempt += 1
                    if attempt <= max_attempts:
                        time.sleep(5)
                    continue

                result = result_response.json()
                status = result.get("status")

                if Status(status) == Status.READY:
                    pbar.update(max_attempts - attempt + 1)  # fill to 100%
                    print(f"[BFL Flux2Klein9bDirect] Ready after {elapsed:.1f}s")
                    return self.process_result(result, output_format=output_format)
                elif Status(status) == Status.PENDING:
                    pbar.update(1)
                    attempt += 1
                    if attempt <= max_attempts:
                        time.sleep(5)
                elif Status(status) in [Status.ERROR, Status.CONTENT_MODERATED, Status.REQUEST_MODERATED]:
                    pbar.update(max_attempts - attempt + 1)
                    print(f"[BFL Flux2Klein9bDirect] Terminal status: {status}")
                    break
                else:
                    pbar.update(1)
                    attempt += 1
                    if attempt <= max_attempts:
                        time.sleep(5)

            except Exception as e:
                print(f"[BFL Flux2Klein9bDirect] Error on poll {attempt}: {str(e)}")
                pbar.update(1)
                attempt += 1
                if attempt <= max_attempts:
                    time.sleep(5)

        print(f"[BFL Flux2Klein9bDirect] Exhausted {max_attempts} attempts — blank image")
        return self.create_blank_image()


NODE_CLASS_MAPPINGS = {
    "Flux2Klein9bDirect_BFL": Flux2Klein9bDirect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2Klein9bDirect_BFL": "Flux 2 Klein 9B Direct (BFL)",
}
