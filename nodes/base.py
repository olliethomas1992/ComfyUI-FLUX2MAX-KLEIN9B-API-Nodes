import requests
from PIL import Image
import io
import numpy as np
import torch
from .config_node import get_config_loader

REQUEST_TIMEOUT = 300  # seconds for connect + read


class BaseFlux:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "BFL"

    def process_result(self, result, output_format="jpeg"):
        try:
            sample_url = result["result"]["sample"]
            img_response = requests.get(sample_url, timeout=REQUEST_TIMEOUT)
            img = Image.open(io.BytesIO(img_response.content))

            with io.BytesIO() as output:
                img.save(output, format=output_format.upper())
                output.seek(0)
                img_converted = Image.open(output)

                img_array = np.array(img_converted).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array)[None,]
                return (img_tensor,)
        except KeyError as e:
            print(f"[BFL] KeyError: Missing expected key {e}")
            return self.create_blank_image()
        except Exception as e:
            print(f"[BFL] Error processing image result: {str(e)}")
            return self.create_blank_image()

    def create_blank_image(self):
        blank_img = Image.new("RGB", (512, 512), color="black")
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return (img_tensor,)

    def post_request(self, url_path, arguments, config_override=None):
        config_loader_instance = get_config_loader(config_override)
        config_loader_instance.set_x_key()

        post_url = config_loader_instance.create_url(url_path)
        headers = {"x-key": config_loader_instance.get_x_key()}

        response = requests.post(
            post_url, json=arguments, headers=headers, timeout=REQUEST_TIMEOUT
        )
        print(f"[BFL] POST {url_path} → {response.status_code}")

        if response.status_code == 200:
            task_id = response.json().get("id")
            print(f"[BFL] Task ID: {task_id}")
            return task_id
        else:
            print(f"[BFL] Error: {response.status_code}, {response.text}")
            return None
