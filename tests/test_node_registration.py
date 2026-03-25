"""Tests that all V3 node classes are properly structured."""
import pytest


# ---------------------------------------------------------------------------
# Flux2Inputs models (8 images + disable_pup): Max, Pro, Pro Preview
# ---------------------------------------------------------------------------

class TestFlux2InputsNodes:
    """Nodes using the Flux2Inputs schema: 8 images, disable_pup, transparent_bg."""

    NODES = [
        ("nodes.flux2max_direct", "Flux2Max", "Flux2Max_BFL", "FLUX.2 [Max] (BFL)", "BFL/FLUX.2"),
        ("nodes.flux2pro", "Flux2Pro", "Flux2Pro_BFL", "FLUX.2 [Pro] (BFL)", "BFL/FLUX.2"),
        ("nodes.flux2pro_preview", "Flux2ProPreview", "Flux2ProPreview_BFL", "FLUX.2 [Pro] Preview (BFL)", "BFL/FLUX.2"),
    ]

    @pytest.mark.parametrize("module,cls_name,node_id,display_name,category", NODES)
    def test_schema_metadata(self, module, cls_name, node_id, display_name, category):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        schema = cls.define_schema()
        assert schema.node_id == node_id
        assert schema.display_name == display_name
        assert schema.category == category

    @pytest.mark.parametrize("module,cls_name,node_id,display_name,category", NODES)
    def test_has_8_image_inputs(self, module, cls_name, node_id, display_name, category):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        schema = cls.define_schema()
        image_inputs = [i for i in schema.inputs if hasattr(i, "id") and i.id.startswith("image_")]
        assert len(image_inputs) == 8

    @pytest.mark.parametrize("module,cls_name,node_id,display_name,category", NODES)
    def test_has_disable_pup(self, module, cls_name, node_id, display_name, category):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        schema = cls.define_schema()
        pup_inputs = [i for i in schema.inputs if hasattr(i, "id") and i.id == "disable_pup"]
        assert len(pup_inputs) == 1

    @pytest.mark.parametrize("module,cls_name,node_id,display_name,category", NODES)
    def test_has_transparent_bg(self, module, cls_name, node_id, display_name, category):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        schema = cls.define_schema()
        tbg_inputs = [i for i in schema.inputs if hasattr(i, "id") and i.id == "transparent_bg"]
        assert len(tbg_inputs) == 1

    @pytest.mark.parametrize("module,cls_name,node_id,display_name,category", NODES)
    def test_execute_is_classmethod(self, module, cls_name, node_id, display_name, category):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        assert isinstance(cls.__dict__["execute"], classmethod)


# ---------------------------------------------------------------------------
# Flux2KleinInputs models (4 images, no disable_pup): Klein 9B, Klein 4B, Klein 9B KV
# ---------------------------------------------------------------------------

class TestFlux2KleinInputsNodes:
    """Nodes using the Flux2KleinInputs schema: 4 images, no disable_pup, transparent_bg."""

    NODES = [
        ("nodes.flux2klein_direct", "Flux2Klein9B", "Flux2Klein9B_BFL", "FLUX.2 [Klein 9B] (BFL)", "BFL/FLUX.2"),
        ("nodes.flux2klein4b", "Flux2Klein4B", "Flux2Klein4B_BFL", "FLUX.2 [Klein 4B] (BFL)", "BFL/FLUX.2"),
        ("nodes.flux2klein9b_kv", "Flux2Klein9BKV", "Flux2Klein9BKV_BFL", "FLUX.2 [Klein 9B KV] (BFL)", "BFL/FLUX.2"),
    ]

    @pytest.mark.parametrize("module,cls_name,node_id,display_name,category", NODES)
    def test_schema_metadata(self, module, cls_name, node_id, display_name, category):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        schema = cls.define_schema()
        assert schema.node_id == node_id
        assert schema.display_name == display_name
        assert schema.category == category

    @pytest.mark.parametrize("module,cls_name,node_id,display_name,category", NODES)
    def test_has_4_image_inputs(self, module, cls_name, node_id, display_name, category):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        schema = cls.define_schema()
        image_inputs = [i for i in schema.inputs if hasattr(i, "id") and i.id.startswith("image_")]
        assert len(image_inputs) == 4

    @pytest.mark.parametrize("module,cls_name,node_id,display_name,category", NODES)
    def test_no_disable_pup(self, module, cls_name, node_id, display_name, category):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        schema = cls.define_schema()
        pup_inputs = [i for i in schema.inputs if hasattr(i, "id") and i.id == "disable_pup"]
        assert len(pup_inputs) == 0

    @pytest.mark.parametrize("module,cls_name,node_id,display_name,category", NODES)
    def test_has_transparent_bg(self, module, cls_name, node_id, display_name, category):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        schema = cls.define_schema()
        tbg_inputs = [i for i in schema.inputs if hasattr(i, "id") and i.id == "transparent_bg"]
        assert len(tbg_inputs) == 1

    @pytest.mark.parametrize("module,cls_name,node_id,display_name,category", NODES)
    def test_execute_is_classmethod(self, module, cls_name, node_id, display_name, category):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        assert isinstance(cls.__dict__["execute"], classmethod)


# ---------------------------------------------------------------------------
# Config node
# ---------------------------------------------------------------------------

def test_config_node_has_schema():
    from nodes.config_node import FluxConfig
    schema = FluxConfig.define_schema()
    assert schema.node_id == "FluxConfig_BFL"
    assert schema.display_name == "Flux Config (BFL)"
    assert schema.category == "BFL/Config"


def test_all_nodes_have_config_input():
    """Every generation node should accept an optional BFL_CONFIG input."""
    import importlib
    all_nodes = [
        ("nodes.flux2max_direct", "Flux2Max"),
        ("nodes.flux2pro", "Flux2Pro"),
        ("nodes.flux2pro_preview", "Flux2ProPreview"),
        ("nodes.flux2klein_direct", "Flux2Klein9B"),
        ("nodes.flux2klein4b", "Flux2Klein4B"),
        ("nodes.flux2klein9b_kv", "Flux2Klein9BKV"),
    ]
    for module, cls_name in all_nodes:
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        schema = cls.define_schema()
        config_inputs = [i for i in schema.inputs if hasattr(i, "id") and i.id == "config"]
        assert len(config_inputs) == 1, f"{cls_name} missing config input"
