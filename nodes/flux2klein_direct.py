"""Flux 2 Klein 9B Direct — V3 async node with 4 IMAGE slots."""
from comfy_api.latest import ComfyExtension, io
import comfy.utils
from .base import image_to_base64, create_blank_image, post_request, poll_for_result

# Custom type for BFL config passthrough
BflConfig = io.Custom("BFL_CONFIG")


class Flux2Klein9bDirect(io.ComfyNode):
    """Flux 2 Klein 9B with direct IMAGE inputs — up to 4 reference images."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Flux2Klein9bDirect_BFL",
            display_name="Flux 2 Klein 9B Direct (BFL)",
            category="BFL/Flux2",
            description="Generate images via Flux 2 Klein 9B API with up to 4 reference images",
            inputs=[
                io.String.Input("prompt", default="", multiline=True),
                io.Int.Input("safety_tolerance", default=2, min=0, max=5),
                io.Combo.Input("output_format", options=["jpeg", "png"], default="jpeg"),
                io.Image.Input("image_1", optional=True),
                io.Image.Input("image_2", optional=True),
                io.Image.Input("image_3", optional=True),
                io.Image.Input("image_4", optional=True),
                io.Int.Input("width", default=0, min=0),
                io.Int.Input("height", default=0, min=0),
                io.Int.Input("seed", default=-1),
                io.String.Input("webhook_url", default="", optional=True),
                io.String.Input("webhook_secret", default="", optional=True),
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

        # Convert IMAGE tensors to base64
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
            task_id = await post_request("flux-2-klein-9b", arguments, config)
            if task_id:
                print(f"[BFL Flux2Klein9bDirect] Task ID '{task_id}'")
                pbar = comfy.utils.ProgressBar(40)
                result = await poll_for_result(
                    task_id,
                    output_format=output_format,
                    config_override=config,
                    pbar=pbar,
                )
                return io.NodeOutput(result)
            return io.NodeOutput(create_blank_image())
        except Exception as e:
            print(f"[BFL Flux2Klein9bDirect] Error: {str(e)}")
            return io.NodeOutput(create_blank_image())
