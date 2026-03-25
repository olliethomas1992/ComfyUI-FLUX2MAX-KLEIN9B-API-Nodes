"""Tests that all V3 node classes are properly structured."""
import pytest


# ---------------------------------------------------------------------------
# Shared seed validation helper
# ---------------------------------------------------------------------------

def _get_seed_input(schema):
    """Extract seed input from schema."""
    seeds = [i for i in schema.inputs if hasattr(i, "id") and i.id == "seed"]
    assert len(seeds) == 1, "Expected exactly one seed input"
    return seeds[0]


def _assert_seed_control(schema):
    """Verify seed has control_after_generate=True."""
    seed = _get_seed_input(schema)
    assert seed.kwargs.get("control_after_generate") is True, "seed missing control_after_generate=True"
    assert seed.kwargs.get("min") == 0
    assert seed.kwargs.get("max") == 0xFFFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# Flux2Inputs models (8 images + prompt_upsampling): Max, Pro, Pro Preview
# ---------------------------------------------------------------------------

class TestFlux2InputsNodes:
    """Nodes using the Flux2Inputs schema: 8 images, prompt_upsampling, transparent_bg."""

    NODES = [
        ("nodes.flux2max_direct", "Flux2Max", "Flux2Max_BFL", "FLUX.2 [Max]", "FLUX.2"),
        ("nodes.flux2pro", "Flux2Pro", "Flux2Pro_BFL", "FLUX.2 [Pro]", "FLUX.2"),
        ("nodes.flux2pro_preview", "Flux2ProPreview", "Flux2ProPreview_BFL", "FLUX.2 [Pro] Preview", "FLUX.2"),
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
    def test_has_prompt_upsampling(self, module, cls_name, node_id, display_name, category):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        schema = cls.define_schema()
        pu_inputs = [i for i in schema.inputs if hasattr(i, "id") and i.id == "prompt_upsampling"]
        assert len(pu_inputs) == 1

    @pytest.mark.parametrize("module,cls_name,node_id,display_name,category", NODES)
    def test_has_transparent_bg(self, module, cls_name, node_id, display_name, category):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        schema = cls.define_schema()
        tbg_inputs = [i for i in schema.inputs if hasattr(i, "id") and i.id == "transparent_bg"]
        assert len(tbg_inputs) == 1

    @pytest.mark.parametrize("module,cls_name,node_id,display_name,category", NODES)
    def test_seed_has_control_after_generate(self, module, cls_name, node_id, display_name, category):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        _assert_seed_control(cls.define_schema())

    @pytest.mark.parametrize("module,cls_name,node_id,display_name,category", NODES)
    def test_execute_is_classmethod(self, module, cls_name, node_id, display_name, category):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        assert isinstance(cls.__dict__["execute"], classmethod)


# ---------------------------------------------------------------------------
# Flux2KleinInputs models (4 images, no prompt_upsampling): Klein 9B, Klein 4B, Klein 9B KV
# ---------------------------------------------------------------------------

class TestFlux2KleinInputsNodes:
    """Nodes using the Flux2KleinInputs schema: 4 images, no prompt_upsampling, transparent_bg."""

    NODES = [
        ("nodes.flux2klein_direct", "Flux2Klein9B", "Flux2Klein9B_BFL", "FLUX.2 [Klein 9B]", "FLUX.2"),
        ("nodes.flux2klein4b", "Flux2Klein4B", "Flux2Klein4B_BFL", "FLUX.2 [Klein 4B]", "FLUX.2"),
        ("nodes.flux2klein9b_kv", "Flux2Klein9BKV", "Flux2Klein9BKV_BFL", "FLUX.2 [Klein 9B KV]", "FLUX.2"),
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
    def test_no_prompt_upsampling(self, module, cls_name, node_id, display_name, category):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        schema = cls.define_schema()
        pu_inputs = [i for i in schema.inputs if hasattr(i, "id") and i.id == "prompt_upsampling"]
        assert len(pu_inputs) == 0

    @pytest.mark.parametrize("module,cls_name,node_id,display_name,category", NODES)
    def test_has_transparent_bg(self, module, cls_name, node_id, display_name, category):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        schema = cls.define_schema()
        tbg_inputs = [i for i in schema.inputs if hasattr(i, "id") and i.id == "transparent_bg"]
        assert len(tbg_inputs) == 1

    @pytest.mark.parametrize("module,cls_name,node_id,display_name,category", NODES)
    def test_seed_has_control_after_generate(self, module, cls_name, node_id, display_name, category):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        _assert_seed_control(cls.define_schema())

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
    assert schema.display_name == "FLUX.2 Config"
    assert schema.category == "FLUX.2"


# ---------------------------------------------------------------------------
# Flux2FlexInputs model: Flex (8 images, guidance, steps, prompt_upsampling)
# ---------------------------------------------------------------------------

class TestFlux2FlexNode:
    """FLUX.2 [Flex] has its own schema: prompt_upsampling, guidance, steps, no transparent_bg."""

    def test_schema_metadata(self):
        from nodes.flux2flex import Flux2Flex
        schema = Flux2Flex.define_schema()
        assert schema.node_id == "Flux2Flex_BFL"
        assert schema.display_name == "FLUX.2 [Flex]"
        assert schema.category == "FLUX.2"

    def test_has_8_image_inputs(self):
        from nodes.flux2flex import Flux2Flex
        schema = Flux2Flex.define_schema()
        image_inputs = [i for i in schema.inputs if hasattr(i, "id") and i.id.startswith("image_")]
        assert len(image_inputs) == 8

    def test_has_prompt_upsampling(self):
        from nodes.flux2flex import Flux2Flex
        schema = Flux2Flex.define_schema()
        pu = [i for i in schema.inputs if hasattr(i, "id") and i.id == "prompt_upsampling"]
        assert len(pu) == 1

    def test_has_guidance(self):
        from nodes.flux2flex import Flux2Flex
        schema = Flux2Flex.define_schema()
        g = [i for i in schema.inputs if hasattr(i, "id") and i.id == "guidance"]
        assert len(g) == 1

    def test_has_steps(self):
        from nodes.flux2flex import Flux2Flex
        schema = Flux2Flex.define_schema()
        s = [i for i in schema.inputs if hasattr(i, "id") and i.id == "steps"]
        assert len(s) == 1

    def test_seed_has_control_after_generate(self):
        from nodes.flux2flex import Flux2Flex
        _assert_seed_control(Flux2Flex.define_schema())

    def test_no_disable_pup_in_api_payload(self):
        """Flex uses prompt_upsampling natively, not disable_pup."""
        from nodes.flux2flex import Flux2Flex
        schema = Flux2Flex.define_schema()
        dp = [i for i in schema.inputs if hasattr(i, "id") and i.id == "disable_pup"]
        assert len(dp) == 0

    def test_no_transparent_bg(self):
        from nodes.flux2flex import Flux2Flex
        schema = Flux2Flex.define_schema()
        tbg = [i for i in schema.inputs if hasattr(i, "id") and i.id == "transparent_bg"]
        assert len(tbg) == 0

    def test_execute_is_classmethod(self):
        from nodes.flux2flex import Flux2Flex
        assert isinstance(Flux2Flex.__dict__["execute"], classmethod)


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
        ("nodes.flux2flex", "Flux2Flex"),
    ]
    for module, cls_name in all_nodes:
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        schema = cls.define_schema()
        config_inputs = [i for i in schema.inputs if hasattr(i, "id") and i.id == "config"]
        assert len(config_inputs) == 1, f"{cls_name} missing config input"
