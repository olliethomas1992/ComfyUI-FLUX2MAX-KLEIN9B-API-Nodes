"""Tests that multiple Direct node instances can run concurrently
without cross-contamination of state, payloads, or API calls."""

import threading
import time
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from nodes.flux2max_direct import Flux2MaxDirect
from nodes.flux2klein_direct import Flux2Klein9bDirect


def _make_image(r, g, b, size=64):
    """Create a solid-colour IMAGE tensor identifiable by its pixel values."""
    arr = np.full((size, size, 3), [r / 255.0, g / 255.0, b / 255.0], dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


# ---------------------------------------------------------------------------
# 1. No shared mutable class state
# ---------------------------------------------------------------------------

class TestNoSharedClassState:
    """Verify node classes have no mutable class-level attributes that could
    leak between instances."""

    def test_flux2max_no_class_dict_or_list(self):
        cls_attrs = {
            k: v for k, v in vars(Flux2MaxDirect).items()
            if not k.startswith("_") and isinstance(v, (dict, list, set))
        }
        assert cls_attrs == {}, f"Mutable class attrs found: {list(cls_attrs.keys())}"

    def test_flux2klein_no_class_dict_or_list(self):
        cls_attrs = {
            k: v for k, v in vars(Flux2Klein9bDirect).items()
            if not k.startswith("_") and isinstance(v, (dict, list, set))
        }
        assert cls_attrs == {}, f"Mutable class attrs found: {list(cls_attrs.keys())}"

    def test_separate_instances_have_no_shared_instance_state(self):
        a = Flux2MaxDirect()
        b = Flux2MaxDirect()
        # Ensure they don't share __dict__
        a.__dict__["_test_marker"] = "instance_a"
        assert "_test_marker" not in b.__dict__


# ---------------------------------------------------------------------------
# 2. Payload isolation — different images produce different API payloads
# ---------------------------------------------------------------------------

class TestPayloadIsolation:
    """Two nodes with different images must build completely independent
    API payloads with no cross-contamination."""

    @patch("nodes.flux2max_direct.Flux2MaxDirect._poll_with_progress")
    @patch("nodes.flux2max_direct.Flux2MaxDirect.post_request")
    def test_two_max_instances_different_images(self, mock_post, mock_poll):
        mock_post.return_value = "task-123"
        blank = torch.zeros(1, 512, 512, 3)
        mock_poll.return_value = (blank,)

        red_img = _make_image(255, 0, 0)
        blue_img = _make_image(0, 0, 255)

        node_a = Flux2MaxDirect()
        node_b = Flux2MaxDirect()

        node_a.generate_image("prompt A", 2, "jpeg", image_1=red_img)
        node_b.generate_image("prompt B", 2, "jpeg", image_1=blue_img)

        assert mock_post.call_count == 2

        args_a = mock_post.call_args_list[0][0][1]
        args_b = mock_post.call_args_list[1][0][1]

        # Prompts must not cross
        assert args_a["prompt"] == "prompt A"
        assert args_b["prompt"] == "prompt B"

        # Base64 payloads must differ (different colours)
        assert args_a["input_image"] != args_b["input_image"]

    @patch("nodes.flux2klein_direct.Flux2Klein9bDirect._poll_with_progress")
    @patch("nodes.flux2klein_direct.Flux2Klein9bDirect.post_request")
    def test_two_klein_instances_different_images(self, mock_post, mock_poll):
        mock_post.return_value = "task-456"
        blank = torch.zeros(1, 512, 512, 3)
        mock_poll.return_value = (blank,)

        green_img = _make_image(0, 255, 0)
        white_img = _make_image(255, 255, 255)

        node_a = Flux2Klein9bDirect()
        node_b = Flux2Klein9bDirect()

        node_a.generate_image("prompt C", 2, "jpeg", image_1=green_img)
        node_b.generate_image("prompt D", 2, "jpeg", image_1=white_img)

        args_a = mock_post.call_args_list[0][0][1]
        args_b = mock_post.call_args_list[1][0][1]

        assert args_a["prompt"] == "prompt C"
        assert args_b["prompt"] == "prompt D"
        assert args_a["input_image"] != args_b["input_image"]


# ---------------------------------------------------------------------------
# 3. Concurrent execution — threaded runs don't interfere
# ---------------------------------------------------------------------------

class TestConcurrentExecution:
    """Simulate two nodes running simultaneously in threads and verify
    each gets its own correct result."""

    @patch("nodes.flux2max_direct.Flux2MaxDirect._poll_with_progress")
    @patch("nodes.flux2max_direct.Flux2MaxDirect.post_request")
    def test_threaded_max_nodes_no_cross_contamination(self, mock_post, mock_poll):
        # Each call gets a unique task ID
        task_ids = iter(["task-thread-1", "task-thread-2"])
        mock_post.side_effect = lambda *a, **kw: next(task_ids)

        # Return different sized blanks so we can tell them apart
        blank_a = torch.zeros(1, 256, 256, 3)
        blank_b = torch.ones(1, 128, 128, 3)
        results = iter([(blank_a,), (blank_b,)])
        mock_poll.side_effect = lambda *a, **kw: next(results)

        red_img = _make_image(255, 0, 0)
        blue_img = _make_image(0, 0, 255)

        outputs = [None, None]
        errors = [None, None]

        def run_node(idx, prompt, img):
            try:
                node = Flux2MaxDirect()
                outputs[idx] = node.generate_image(prompt, 2, "jpeg", image_1=img)
            except Exception as e:
                errors[idx] = e

        t1 = threading.Thread(target=run_node, args=(0, "thread-prompt-A", red_img))
        t2 = threading.Thread(target=run_node, args=(1, "thread-prompt-B", blue_img))

        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert errors[0] is None, f"Thread 1 error: {errors[0]}"
        assert errors[1] is None, f"Thread 2 error: {errors[1]}"
        assert outputs[0] is not None
        assert outputs[1] is not None

        # Both post_request calls happened with different prompts
        prompts_sent = [call[0][1]["prompt"] for call in mock_post.call_args_list]
        assert "thread-prompt-A" in prompts_sent
        assert "thread-prompt-B" in prompts_sent

    @patch("nodes.flux2klein_direct.Flux2Klein9bDirect._poll_with_progress")
    @patch("nodes.flux2klein_direct.Flux2Klein9bDirect.post_request")
    def test_threaded_klein_nodes_no_cross_contamination(self, mock_post, mock_poll):
        task_ids = iter(["task-klein-1", "task-klein-2"])
        mock_post.side_effect = lambda *a, **kw: next(task_ids)

        blank = torch.zeros(1, 512, 512, 3)
        mock_poll.side_effect = lambda *a, **kw: (blank,)

        outputs = [None, None]
        errors = [None, None]

        def run_node(idx, prompt):
            try:
                node = Flux2Klein9bDirect()
                outputs[idx] = node.generate_image(prompt, 2, "jpeg", image_1=_make_image(idx * 100, 0, 0))
            except Exception as e:
                errors[idx] = e

        t1 = threading.Thread(target=run_node, args=(0, "klein-A"))
        t2 = threading.Thread(target=run_node, args=(1, "klein-B"))

        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert errors[0] is None
        assert errors[1] is None
        prompts_sent = [call[0][1]["prompt"] for call in mock_post.call_args_list]
        assert "klein-A" in prompts_sent
        assert "klein-B" in prompts_sent


# ---------------------------------------------------------------------------
# 4. Mixed node types — Max and Klein running together
# ---------------------------------------------------------------------------

class TestMixedNodeTypes:
    """Max and Klein nodes running concurrently must not share any state."""

    @patch("nodes.flux2klein_direct.Flux2Klein9bDirect._poll_with_progress")
    @patch("nodes.flux2klein_direct.Flux2Klein9bDirect.post_request")
    @patch("nodes.flux2max_direct.Flux2MaxDirect._poll_with_progress")
    @patch("nodes.flux2max_direct.Flux2MaxDirect.post_request")
    def test_max_and_klein_concurrent(self, mock_max_post, mock_max_poll,
                                      mock_klein_post, mock_klein_poll):
        mock_max_post.return_value = "task-max"
        mock_klein_post.return_value = "task-klein"
        blank = torch.zeros(1, 512, 512, 3)
        mock_max_poll.return_value = (blank,)
        mock_klein_poll.return_value = (blank,)

        outputs = [None, None]
        errors = [None, None]

        def run_max():
            try:
                node = Flux2MaxDirect()
                outputs[0] = node.generate_image("max-prompt", 2, "jpeg",
                                                  image_1=_make_image(255, 0, 0))
            except Exception as e:
                errors[0] = e

        def run_klein():
            try:
                node = Flux2Klein9bDirect()
                outputs[1] = node.generate_image("klein-prompt", 2, "jpeg",
                                                  image_1=_make_image(0, 255, 0))
            except Exception as e:
                errors[1] = e

        t1 = threading.Thread(target=run_max)
        t2 = threading.Thread(target=run_klein)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert errors[0] is None
        assert errors[1] is None

        # Max node called flux-2-max endpoint
        max_args = mock_max_post.call_args_list[0]
        assert max_args[0][0] == "flux-2-max"
        assert max_args[0][1]["prompt"] == "max-prompt"

        # Klein node called flux-2-klein-9b endpoint
        klein_args = mock_klein_post.call_args_list[0]
        assert klein_args[0][0] == "flux-2-klein-9b"
        assert klein_args[0][1]["prompt"] == "klein-prompt"
