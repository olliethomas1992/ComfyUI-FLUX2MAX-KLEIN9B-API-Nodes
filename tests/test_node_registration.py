"""Tests that V3 node classes are properly structured."""
import pytest


def test_flux2max_direct_has_schema():
    from nodes.flux2max_direct import Flux2MaxDirect

    schema = Flux2MaxDirect.define_schema()
    assert schema.node_id == "Flux2MaxDirect_BFL"
    assert schema.display_name == "Flux 2 Max Direct (BFL)"
    assert schema.category == "BFL/Flux2"


def test_flux2klein_direct_has_schema():
    from nodes.flux2klein_direct import Flux2Klein9bDirect

    schema = Flux2Klein9bDirect.define_schema()
    assert schema.node_id == "Flux2Klein9bDirect_BFL"
    assert schema.display_name == "Flux 2 Klein 9B Direct (BFL)"
    assert schema.category == "BFL/Flux2"


def test_flux2max_has_8_image_inputs():
    from nodes.flux2max_direct import Flux2MaxDirect

    schema = Flux2MaxDirect.define_schema()
    image_inputs = [i for i in schema.inputs if hasattr(i, 'id') and i.id.startswith("image_")]
    assert len(image_inputs) == 8


def test_flux2klein_has_4_image_inputs():
    from nodes.flux2klein_direct import Flux2Klein9bDirect

    schema = Flux2Klein9bDirect.define_schema()
    image_inputs = [i for i in schema.inputs if hasattr(i, 'id') and i.id.startswith("image_")]
    assert len(image_inputs) == 4


def test_flux2max_has_config_input():
    from nodes.flux2max_direct import Flux2MaxDirect

    schema = Flux2MaxDirect.define_schema()
    config_inputs = [i for i in schema.inputs if hasattr(i, 'id') and i.id == "config"]
    assert len(config_inputs) == 1


def test_flux2klein_has_config_input():
    from nodes.flux2klein_direct import Flux2Klein9bDirect

    schema = Flux2Klein9bDirect.define_schema()
    config_inputs = [i for i in schema.inputs if hasattr(i, 'id') and i.id == "config"]
    assert len(config_inputs) == 1


def test_execute_is_classmethod():
    from nodes.flux2max_direct import Flux2MaxDirect
    from nodes.flux2klein_direct import Flux2Klein9bDirect

    assert isinstance(
        Flux2MaxDirect.__dict__["execute"], classmethod
    )
    assert isinstance(
        Flux2Klein9bDirect.__dict__["execute"], classmethod
    )


def test_config_node_has_schema():
    from nodes.config_node import FluxConfig

    schema = FluxConfig.define_schema()
    assert schema.node_id == "FluxConfig_BFL"
    assert schema.display_name == "Flux Config (BFL)"
    assert schema.category == "BFL/Config"
