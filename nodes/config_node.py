"""BFL Config node — V3 schema."""
from comfy_api.latest import io
from .config import ConfigLoader

# Custom type for BFL config
BflConfig = io.Custom("BFL_CONFIG")


class FluxConfig(io.ComfyNode):
    """Configuration node for BFL API settings.
    Provides optional configuration that can be connected to other nodes.
    If not connected, nodes will use the default file-based configuration."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FluxConfig_BFL",
            display_name="FLUX.2 Config",
            category="FLUX.2",
            description="Configure BFL API key and endpoint for Flux nodes",
            inputs=[
                io.String.Input("x_key", default="", tooltip="BFL API key"),
                io.String.Input(
                    "base_url",
                    default="https://api.bfl.ai/v1/",
                    tooltip="Base API URL",
                ),
                io.Combo.Input(
                    "region",
                    options=["none", "us", "eu"],
                    default="none",
                    tooltip="Regional endpoint for finetuning operations",
                    optional=True,
                ),
            ],
            outputs=[
                BflConfig.Output(display_name="config"),
            ],
        )

    @classmethod
    def execute(cls, x_key, base_url, region="none"):
        regional_endpoints = {
            "us": "https://api.us.bfl.ai",
            "eu": "https://api.eu.bfl.ai",
        }

        config = {
            "x_key": x_key.strip() if x_key.strip() else None,
            "base_url": base_url.strip() if base_url.strip() else "https://api.bfl.ai/v1/",
            "regional_endpoints": regional_endpoints,
            "default_region": region if region != "none" else None,
        }

        return io.NodeOutput(config)


def get_config_loader(config_override=None):
    """Get a ConfigLoader instance with optional config override."""
    return ConfigLoader(config_override)
