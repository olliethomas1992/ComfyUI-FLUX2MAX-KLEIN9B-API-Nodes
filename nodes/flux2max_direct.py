"""FLUX.2 [Max] — V3 async node with 8 IMAGE slots."""
from comfy_api.latest import io
import comfy.utils
from .base import image_to_base64, create_blank_image, post_request, poll_for_result

BflConfig = io.Custom("BFL_CONFIG")


class Flux2Max(io.ComfyNode):
    """Generate images via FLUX.2 [Max] API with up to 8 reference images."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Flux2Max_BFL",
            display_name="FLUX.2 [Max]",
            category="FLUX.2",
            description="Generate images via FLUX.2 [Max] API with up to 8 reference images",
            inputs=[
                io.String.Input("prompt", default="", multiline=True, tooltip="Text prompt describing the desired image"),
                io.Boolean.Input("prompt_upsampling", default=True, tooltip="Enhance your prompt with AI-powered upsampling for better results"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True, tooltip="Seed for reproducible results"),
                io.Int.Input("width", default=1024, min=0, max=4096, step=32, tooltip="Image width in pixels (0 = auto, min 64, must be multiple of 32)"),
                io.Int.Input("height", default=1024, min=0, max=4096, step=32, tooltip="Image height in pixels (0 = auto, min 64, must be multiple of 32)"),
                io.Int.Input("safety_tolerance", default=2, min=0, max=5, tooltip="Content filter strictness (0 = most strict, 5 = most lenient)"),
                io.Combo.Input("output_format", options=["jpeg", "png", "webp"], default="jpeg", tooltip="Output image format"),
                io.Boolean.Input("transparent_bg", default=False, tooltip="Remove background and return RGBA PNG"),
                io.Image.Input("image_1", optional=True, tooltip="Reference image 1"),
                io.Image.Input("image_2", optional=True, tooltip="Reference image 2"),
                io.Image.Input("image_3", optional=True, tooltip="Reference image 3"),
                io.Image.Input("image_4", optional=True, tooltip="Reference image 4"),
                io.Image.Input("image_5", optional=True, tooltip="Reference image 5"),
                io.Image.Input("image_6", optional=True, tooltip="Reference image 6"),
                io.Image.Input("image_7", optional=True, tooltip="Reference image 7"),
                io.Image.Input("image_8", optional=True, tooltip="Reference image 8"),
                io.String.Input("webhook_url", default="", optional=True, tooltip="Optional webhook URL for async notifications"),
                io.String.Input("webhook_secret", default="", optional=True, tooltip="Optional webhook secret for verification"),
                BflConfig.Input("config", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        prompt,
        prompt_upsampling,
        seed,
        width,
        height,
        safety_tolerance,
        output_format,
        transparent_bg,
        image_1=None, image_2=None, image_3=None, image_4=None,
        image_5=None, image_6=None, image_7=None, image_8=None,
        webhook_url="", webhook_secret="",
        config=None,
    ):
        arguments = {
            "prompt": prompt,
            "disable_pup": not prompt_upsampling,
            "seed": seed,
            "safety_tolerance": safety_tolerance,
            "output_format": output_format,
            "transparent_bg": transparent_bg,
        }

        image_slots = [
            ("input_image", image_1), ("input_image_2", image_2),
            ("input_image_3", image_3), ("input_image_4", image_4),
            ("input_image_5", image_5), ("input_image_6", image_6),
            ("input_image_7", image_7), ("input_image_8", image_8),
        ]
        for key, img in image_slots:
            if img is not None:
                arguments[key] = image_to_base64(img)

        if width > 0:
            arguments["width"] = width
        if height > 0:
            arguments["height"] = height
        if webhook_url:
            arguments["webhook_url"] = webhook_url
        if webhook_secret:
            arguments["webhook_secret"] = webhook_secret

        try:
            task_id = await post_request("flux-2-max", arguments, config)
            if task_id:
                print(f"[BFL FLUX.2 Max] Task ID '{task_id}'")
                pbar = comfy.utils.ProgressBar(40)
                result = await poll_for_result(task_id, output_format=output_format, config_override=config, pbar=pbar)
                return io.NodeOutput(result)
            return io.NodeOutput(create_blank_image())
        except Exception as e:
            print(f"[BFL FLUX.2 Max] Error: {str(e)}")
            return io.NodeOutput(create_blank_image())
