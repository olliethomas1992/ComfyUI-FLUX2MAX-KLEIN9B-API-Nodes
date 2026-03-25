"""ComfyUI V3 node pack for all FLUX.2 models (BFL API)."""
from comfy_api.latest import ComfyExtension
from .nodes.flux2max_direct import Flux2Max
from .nodes.flux2pro import Flux2Pro
from .nodes.flux2pro_preview import Flux2ProPreview
from .nodes.flux2klein_direct import Flux2Klein9B
from .nodes.flux2klein4b import Flux2Klein4B
from .nodes.flux2klein9b_kv import Flux2Klein9BKV
from .nodes.config_node import FluxConfig


class BflFlux2Extension(ComfyExtension):
    async def get_node_list(self):
        return [
            Flux2Max,
            Flux2Pro,
            Flux2ProPreview,
            Flux2Klein9B,
            Flux2Klein4B,
            Flux2Klein9BKV,
            FluxConfig,
        ]


async def comfy_entrypoint():
    return BflFlux2Extension()


WEB_DIRECTORY = "./web"
