"""ComfyUI V3 node pack for Flux 2 Max and Klein 9B (BFL API)."""
from comfy_api.latest import ComfyExtension
from .nodes.flux2max_direct import Flux2MaxDirect
from .nodes.flux2klein_direct import Flux2Klein9bDirect
from .nodes.config_node import FluxConfig


class BflFluxExtension(ComfyExtension):
    async def get_node_list(self):
        return [Flux2MaxDirect, Flux2Klein9bDirect, FluxConfig]


async def comfy_entrypoint():
    return BflFluxExtension()


WEB_DIRECTORY = "./web"
