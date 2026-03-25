"""Tests that V3 stateless classmethods produce isolated payloads
without cross-contamination of state or API calls."""

import asyncio
from unittest.mock import patch, AsyncMock

import numpy as np
import torch

from nodes.flux2max_direct import Flux2Max
from nodes.flux2klein_direct import Flux2Klein9B


def run_async(coro):
    return asyncio.run(coro)


def _make_image(r, g, b, size=64):
    """Create a solid-colour IMAGE tensor identifiable by its pixel values."""
    arr = np.full((size, size, 3), [r / 255.0, g / 255.0, b / 255.0], dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


# ---------------------------------------------------------------------------
# 1. No shared mutable class state
# ---------------------------------------------------------------------------

class TestNoSharedClassState:

    def test_flux2max_no_mutable_class_attrs(self):
        cls_attrs = {
            k: v for k, v in vars(Flux2Max).items()
            if not k.startswith("_") and isinstance(v, (dict, list, set))
        }
        assert cls_attrs == {}, f"Mutable class attrs found: {list(cls_attrs.keys())}"

    def test_flux2klein_no_mutable_class_attrs(self):
        cls_attrs = {
            k: v for k, v in vars(Flux2Klein9B).items()
            if not k.startswith("_") and isinstance(v, (dict, list, set))
        }
        assert cls_attrs == {}, f"Mutable class attrs found: {list(cls_attrs.keys())}"


# ---------------------------------------------------------------------------
# 2. Payload isolation
# ---------------------------------------------------------------------------

class TestPayloadIsolation:

    @patch("nodes.flux2max_direct.poll_for_result", new_callable=AsyncMock, return_value=torch.zeros(1, 512, 512, 3))
    @patch("nodes.flux2max_direct.post_request", new_callable=AsyncMock, return_value="task-123")
    def test_two_max_calls_different_images(self, mock_post, mock_poll):
        red_img = _make_image(255, 0, 0)
        blue_img = _make_image(0, 0, 255)

        run_async(Flux2Max.execute(prompt="prompt A", disable_pup=False, safety_tolerance=2, output_format="jpeg", transparent_bg=False, image_1=red_img))
        run_async(Flux2Max.execute(prompt="prompt B", disable_pup=False, safety_tolerance=2, output_format="jpeg", transparent_bg=False, image_1=blue_img))

        assert mock_post.call_count == 2

        args_a = mock_post.call_args_list[0][0][1]
        args_b = mock_post.call_args_list[1][0][1]

        assert args_a["prompt"] == "prompt A"
        assert args_b["prompt"] == "prompt B"
        assert args_a["input_image"] != args_b["input_image"]

    @patch("nodes.flux2klein_direct.poll_for_result", new_callable=AsyncMock, return_value=torch.zeros(1, 512, 512, 3))
    @patch("nodes.flux2klein_direct.post_request", new_callable=AsyncMock, return_value="task-456")
    def test_two_klein_calls_different_images(self, mock_post, mock_poll):
        green_img = _make_image(0, 255, 0)
        white_img = _make_image(255, 255, 255)

        run_async(Flux2Klein9B.execute(prompt="prompt C", safety_tolerance=2, output_format="jpeg", transparent_bg=False, image_1=green_img))
        run_async(Flux2Klein9B.execute(prompt="prompt D", safety_tolerance=2, output_format="jpeg", transparent_bg=False, image_1=white_img))

        args_a = mock_post.call_args_list[0][0][1]
        args_b = mock_post.call_args_list[1][0][1]

        assert args_a["prompt"] == "prompt C"
        assert args_b["prompt"] == "prompt D"
        assert args_a["input_image"] != args_b["input_image"]


# ---------------------------------------------------------------------------
# 3. Concurrent async execution
# ---------------------------------------------------------------------------

class TestConcurrentExecution:

    @patch("nodes.flux2max_direct.poll_for_result", new_callable=AsyncMock, return_value=torch.zeros(1, 512, 512, 3))
    @patch("nodes.flux2max_direct.post_request", new_callable=AsyncMock)
    def test_concurrent_max_nodes(self, mock_post, mock_poll):
        task_ids = iter(["task-1", "task-2"])
        mock_post.side_effect = lambda *a, **kw: next(task_ids)

        red_img = _make_image(255, 0, 0)
        blue_img = _make_image(0, 0, 255)

        async def run_both():
            r1 = Flux2Max.execute(prompt="async-A", disable_pup=False, safety_tolerance=2, output_format="jpeg", transparent_bg=False, image_1=red_img)
            r2 = Flux2Max.execute(prompt="async-B", disable_pup=False, safety_tolerance=2, output_format="jpeg", transparent_bg=False, image_1=blue_img)
            return await asyncio.gather(r1, r2)

        results = run_async(run_both())

        assert len(results) == 2
        prompts_sent = [call[0][1]["prompt"] for call in mock_post.call_args_list]
        assert "async-A" in prompts_sent
        assert "async-B" in prompts_sent


# ---------------------------------------------------------------------------
# 4. Mixed node types
# ---------------------------------------------------------------------------

class TestMixedNodeTypes:

    @patch("nodes.flux2klein_direct.poll_for_result", new_callable=AsyncMock, return_value=torch.zeros(1, 512, 512, 3))
    @patch("nodes.flux2klein_direct.post_request", new_callable=AsyncMock)
    @patch("nodes.flux2max_direct.poll_for_result", new_callable=AsyncMock, return_value=torch.zeros(1, 512, 512, 3))
    @patch("nodes.flux2max_direct.post_request", new_callable=AsyncMock)
    def test_max_and_klein_concurrent(self, mock_max_post, mock_max_poll, mock_klein_post, mock_klein_poll):
        task_ids_max = iter(["task-max"])
        task_ids_klein = iter(["task-klein"])
        mock_max_post.side_effect = lambda *a, **kw: next(task_ids_max)
        mock_klein_post.side_effect = lambda *a, **kw: next(task_ids_klein)

        async def run_both():
            r1 = Flux2Max.execute(prompt="max-prompt", disable_pup=False, safety_tolerance=2, output_format="jpeg", transparent_bg=False, image_1=_make_image(255, 0, 0))
            r2 = Flux2Klein9B.execute(prompt="klein-prompt", safety_tolerance=2, output_format="jpeg", transparent_bg=False, image_1=_make_image(0, 255, 0))
            return await asyncio.gather(r1, r2)

        results = run_async(run_both())

        assert len(results) == 2

        max_prompts = [c[0][1]["prompt"] for c in mock_max_post.call_args_list]
        klein_prompts = [c[0][1]["prompt"] for c in mock_klein_post.call_args_list]
        assert "max-prompt" in max_prompts
        assert "klein-prompt" in klein_prompts
