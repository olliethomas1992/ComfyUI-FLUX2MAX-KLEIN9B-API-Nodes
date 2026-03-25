"""Shared fixtures and stubs for BFL node tests (V3 schema)."""
import sys
import os
import types

# --- Path setup ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

# --- Stub comfy.utils before any node imports ---
comfy_module = types.ModuleType("comfy")
comfy_utils = types.ModuleType("comfy.utils")


class FakeProgressBar:
    def __init__(self, total):
        self.total = total
        self.updates = []

    def update(self, value):
        self.updates.append(value)


comfy_utils.ProgressBar = FakeProgressBar
comfy_module.utils = comfy_utils
sys.modules["comfy"] = comfy_module
sys.modules["comfy.utils"] = comfy_utils

# --- Stub comfy_api.latest for V3 schema ---
comfy_api_module = types.ModuleType("comfy_api")
comfy_api_latest = types.ModuleType("comfy_api.latest")


class FakeNodeOutput:
    """Mimics io.NodeOutput — stores positional args and kwargs."""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @property
    def result(self):
        return self.args[0] if self.args else None


class FakeInput:
    def __init__(self, id, **kwargs):
        self.id = id
        self.optional = kwargs.get("optional", False)
        self.kwargs = kwargs


class FakeOutput:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeSchema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeComfyNode:
    """Base class stub for io.ComfyNode."""
    pass


class FakeCustom:
    """Stub for io.Custom('TYPE')."""
    def __init__(self, type_name):
        self.type_name = type_name

    def Input(self, id, **kwargs):
        return FakeInput(id, **kwargs)

    def Output(self, **kwargs):
        return FakeOutput(**kwargs)


def _make_type_class():
    """Create a fake type class with Input/Output subclasses."""
    class TypeClass:
        class Input:
            def __init__(self, id, **kwargs):
                self.id = id
                self.kwargs = kwargs
        class Output:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
    return TypeClass


class FakeIO:
    ComfyNode = FakeComfyNode
    Schema = FakeSchema
    NodeOutput = FakeNodeOutput
    Image = _make_type_class()
    Int = _make_type_class()
    Float = _make_type_class()
    String = _make_type_class()
    Boolean = _make_type_class()
    Mask = _make_type_class()

    class Combo:
        class Input:
            def __init__(self, id, options=None, **kwargs):
                self.id = id
                self.options = options
                self.kwargs = kwargs
        class Output:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

    @staticmethod
    def Custom(type_name):
        return FakeCustom(type_name)


class FakeUI:
    pass


class FakeComfyExtension:
    pass


comfy_api_latest.io = FakeIO()
comfy_api_latest.ui = FakeUI()
comfy_api_latest.ComfyExtension = FakeComfyExtension

comfy_api_module.latest = comfy_api_latest
sys.modules["comfy_api"] = comfy_api_module
sys.modules["comfy_api.latest"] = comfy_api_latest

# --- Pre-import the node submodules so @patch targets resolve ---
import nodes.flux2max_direct  # noqa: E402
import nodes.flux2pro  # noqa: E402
import nodes.flux2pro_preview  # noqa: E402
import nodes.flux2klein_direct  # noqa: E402
import nodes.flux2klein4b  # noqa: E402
import nodes.flux2klein9b_kv  # noqa: E402

# --- Fixtures ---
import pytest  # noqa: E402
from helpers import make_image_tensor, make_sample_jpeg_bytes  # noqa: E402


@pytest.fixture
def dummy_image():
    return make_image_tensor()


@pytest.fixture
def sample_jpeg_bytes():
    return make_sample_jpeg_bytes()
