"""Tests for the async polling + progress bar behaviour (V3 schema)."""
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from helpers import (
    make_ready_response,
    make_pending_response,
    make_error_response,
    make_moderated_response,
    make_sample_jpeg_bytes,
)


def run_async(coro):
    return asyncio.run(coro)


class FakeResponse:
    def __init__(self, status, json_data=None, content=None):
        self.status = status
        self._json = json_data
        self._content = content

    async def json(self):
        return self._json

    async def read(self):
        return self._content

    async def text(self):
        return str(self._json)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def _make_fake_session(responses):
    """Build a fake aiohttp.ClientSession that yields responses in order."""
    call_count = [0]

    class FakeSession:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def get(self, url, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx]

        def post(self, url, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx]

    return FakeSession, call_count


@pytest.fixture
def mock_config():
    loader = MagicMock()
    loader.get_x_key.return_value = "test-key"
    loader.create_url.return_value = "https://api.bfl.ai/get_result?id=test-task"
    return loader


# ---------------------------------------------------------------------------
# poll_for_result tests
# ---------------------------------------------------------------------------

class TestPollForResult:

    @patch("nodes.base.asyncio.sleep", new_callable=AsyncMock)
    @patch("nodes.base.get_config_loader")
    def test_ready_on_first_poll(self, mock_gcl, mock_sleep, mock_config, sample_jpeg_bytes):
        mock_gcl.return_value = mock_config

        FakeSession, count = _make_fake_session([
            FakeResponse(200, make_ready_response("https://example.com/img.jpg")),
            FakeResponse(200, content=sample_jpeg_bytes),
        ])

        with patch("nodes.base.aiohttp.ClientSession", FakeSession):
            from nodes.base import poll_for_result
            result = run_async(poll_for_result("test-task"))

        assert result.shape[0] == 1
        assert result.shape[3] == 3
        mock_sleep.assert_not_called()

    @patch("nodes.base.asyncio.sleep", new_callable=AsyncMock)
    @patch("nodes.base.get_config_loader")
    def test_pending_then_ready(self, mock_gcl, mock_sleep, mock_config, sample_jpeg_bytes):
        mock_gcl.return_value = mock_config

        FakeSession, count = _make_fake_session([
            FakeResponse(200, make_pending_response()),
            FakeResponse(200, make_pending_response()),
            FakeResponse(200, make_ready_response()),
            FakeResponse(200, content=sample_jpeg_bytes),
        ])

        with patch("nodes.base.aiohttp.ClientSession", FakeSession):
            from nodes.base import poll_for_result
            result = run_async(poll_for_result("test-task"))

        assert result.shape[3] == 3
        assert mock_sleep.call_count == 2

    @patch("nodes.base.asyncio.sleep", new_callable=AsyncMock)
    @patch("nodes.base.get_config_loader")
    def test_error_stops_polling(self, mock_gcl, mock_sleep, mock_config):
        mock_gcl.return_value = mock_config

        FakeSession, count = _make_fake_session([
            FakeResponse(200, make_error_response()),
        ])

        with patch("nodes.base.aiohttp.ClientSession", FakeSession):
            from nodes.base import poll_for_result
            result = run_async(poll_for_result("test-task"))

        assert result.shape == (1, 512, 512, 3)
        assert count[0] == 1

    @patch("nodes.base.asyncio.sleep", new_callable=AsyncMock)
    @patch("nodes.base.get_config_loader")
    def test_moderated_stops_polling(self, mock_gcl, mock_sleep, mock_config):
        mock_gcl.return_value = mock_config

        FakeSession, count = _make_fake_session([
            FakeResponse(200, make_moderated_response()),
        ])

        with patch("nodes.base.aiohttp.ClientSession", FakeSession):
            from nodes.base import poll_for_result
            result = run_async(poll_for_result("test-task"))

        assert result.shape == (1, 512, 512, 3)
        assert count[0] == 1

    @patch("nodes.base.asyncio.sleep", new_callable=AsyncMock)
    @patch("nodes.base.get_config_loader")
    def test_exhausted_attempts_returns_blank(self, mock_gcl, mock_sleep, mock_config):
        mock_gcl.return_value = mock_config

        # Need enough responses for MAX_POLL_ATTEMPTS
        from nodes.base import MAX_POLL_ATTEMPTS
        FakeSession, count = _make_fake_session(
            [FakeResponse(200, make_pending_response()) for _ in range(MAX_POLL_ATTEMPTS)]
        )

        with patch("nodes.base.aiohttp.ClientSession", FakeSession):
            from nodes.base import poll_for_result
            result = run_async(poll_for_result("test-task"))

        assert result.shape == (1, 512, 512, 3)
        assert count[0] == MAX_POLL_ATTEMPTS

    @patch("nodes.base.asyncio.sleep", new_callable=AsyncMock)
    @patch("nodes.base.get_config_loader")
    def test_http_error_retries(self, mock_gcl, mock_sleep, mock_config, sample_jpeg_bytes):
        mock_gcl.return_value = mock_config

        FakeSession, count = _make_fake_session([
            FakeResponse(500),
            FakeResponse(200, make_ready_response()),
            FakeResponse(200, content=sample_jpeg_bytes),
        ])

        with patch("nodes.base.aiohttp.ClientSession", FakeSession):
            from nodes.base import poll_for_result
            result = run_async(poll_for_result("test-task"))

        assert result.shape[3] == 3


# ---------------------------------------------------------------------------
# Progress bar behaviour
# ---------------------------------------------------------------------------

class TestProgressBar:

    @patch("nodes.base.asyncio.sleep", new_callable=AsyncMock)
    @patch("nodes.base.get_config_loader")
    def test_progress_fills_on_ready(self, mock_gcl, mock_sleep, mock_config, sample_jpeg_bytes):
        mock_gcl.return_value = mock_config
        import comfy.utils

        from nodes.base import MAX_POLL_ATTEMPTS
        pbar = comfy.utils.ProgressBar(MAX_POLL_ATTEMPTS)

        FakeSession, count = _make_fake_session([
            FakeResponse(200, make_pending_response()),
            FakeResponse(200, make_pending_response()),
            FakeResponse(200, make_ready_response()),
            FakeResponse(200, content=sample_jpeg_bytes),
        ])

        with patch("nodes.base.aiohttp.ClientSession", FakeSession):
            from nodes.base import poll_for_result
            result = run_async(poll_for_result("test-task", pbar=pbar))

        assert result.shape[3] == 3
        assert sum(pbar.updates) == MAX_POLL_ATTEMPTS

    @patch("nodes.base.asyncio.sleep", new_callable=AsyncMock)
    @patch("nodes.base.get_config_loader")
    def test_progress_fills_on_error(self, mock_gcl, mock_sleep, mock_config):
        mock_gcl.return_value = mock_config
        import comfy.utils

        from nodes.base import MAX_POLL_ATTEMPTS
        pbar = comfy.utils.ProgressBar(MAX_POLL_ATTEMPTS)

        FakeSession, count = _make_fake_session([
            FakeResponse(200, make_pending_response()),
            FakeResponse(200, make_error_response()),
        ])

        with patch("nodes.base.aiohttp.ClientSession", FakeSession):
            from nodes.base import poll_for_result
            result = run_async(poll_for_result("test-task", pbar=pbar))

        assert result.shape == (1, 512, 512, 3)
