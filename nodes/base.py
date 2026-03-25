"""Shared async utilities for BFL API nodes (V3 schema)."""
import asyncio
import aiohttp
from PIL import Image
import io as stdio
import numpy as np
import torch
from .config_node import get_config_loader

REQUEST_TIMEOUT = 300  # seconds
POLL_INTERVAL = 5  # seconds between polls
MAX_POLL_ATTEMPTS = 40


def image_to_base64(image_tensor):
    """Convert a ComfyUI IMAGE tensor to base64 JPEG string."""
    import base64

    img_array = (image_tensor[0].numpy() * 255).astype(np.uint8)
    pil_image = Image.fromarray(img_array).convert("RGB")
    buffer = stdio.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_blank_image():
    """Return a black 512x512 IMAGE tensor as a fallback."""
    blank_img = Image.new("RGB", (512, 512), color="black")
    img_array = np.array(blank_img).astype(np.float32) / 255.0
    return torch.from_numpy(img_array)[None,]


async def download_and_process_image(sample_url, output_format="jpeg"):
    """Download result image from BFL and convert to IMAGE tensor."""
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(sample_url) as resp:
            data = await resp.read()

    img = Image.open(stdio.BytesIO(data))
    with stdio.BytesIO() as output:
        img.save(output, format=output_format.upper())
        output.seek(0)
        img_converted = Image.open(output)
        img_array = np.array(img_converted).astype(np.float32) / 255.0
        return torch.from_numpy(img_array)[None,]


async def post_request(url_path, arguments, config_override=None):
    """Submit a generation request to the BFL API. Returns task_id or None."""
    config_loader_instance = get_config_loader(config_override)
    config_loader_instance.set_x_key()

    post_url = config_loader_instance.create_url(url_path)
    headers = {"x-key": config_loader_instance.get_x_key()}

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(post_url, json=arguments, headers=headers) as resp:
            status_code = resp.status
            print(f"[BFL] POST {url_path} → {status_code}")

            if status_code == 200:
                result = await resp.json()
                task_id = result.get("id")
                print(f"[BFL] Task ID: {task_id}")
                return task_id
            else:
                text = await resp.text()
                print(f"[BFL] Error: {status_code}, {text}")
                return None


async def poll_for_result(task_id, output_format="jpeg", config_override=None, pbar=None):
    """Poll BFL API for task result. Returns IMAGE tensor or blank image."""
    from .status import Status

    config_loader_instance = get_config_loader(config_override)
    headers = {"x-key": config_loader_instance.get_x_key()}
    get_url = config_loader_instance.create_url(f"get_result?id={task_id}")

    import time
    start_time = time.time()
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        elapsed = time.time() - start_time
        try:
            print(f"[BFL] Poll {attempt}/{MAX_POLL_ATTEMPTS} | {elapsed:.1f}s")

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(get_url, headers=headers) as resp:
                    if resp.status != 200:
                        print(f"[BFL] HTTP {resp.status}")
                        if pbar:
                            pbar.update(1)
                        if attempt < MAX_POLL_ATTEMPTS:
                            await asyncio.sleep(POLL_INTERVAL)
                        continue

                    result = await resp.json()

            status = result.get("status")

            if Status(status) == Status.READY:
                if pbar:
                    pbar.update(MAX_POLL_ATTEMPTS - attempt + 1)
                print(f"[BFL] Ready after {elapsed:.1f}s")
                sample_url = result["result"]["sample"]
                return await download_and_process_image(sample_url, output_format)

            elif Status(status) == Status.PENDING:
                if pbar:
                    pbar.update(1)
                if attempt < MAX_POLL_ATTEMPTS:
                    await asyncio.sleep(POLL_INTERVAL)

            elif Status(status) in [Status.ERROR, Status.CONTENT_MODERATED, Status.REQUEST_MODERATED]:
                if pbar:
                    pbar.update(MAX_POLL_ATTEMPTS - attempt + 1)
                print(f"[BFL] Terminal status: {status}")
                return create_blank_image()

            else:
                if pbar:
                    pbar.update(1)
                if attempt < MAX_POLL_ATTEMPTS:
                    await asyncio.sleep(POLL_INTERVAL)

        except Exception as e:
            print(f"[BFL] Error on poll {attempt}: {str(e)}")
            if pbar:
                pbar.update(1)
            if attempt < MAX_POLL_ATTEMPTS:
                await asyncio.sleep(POLL_INTERVAL)

    print(f"[BFL] Exhausted {MAX_POLL_ATTEMPTS} attempts — blank image")
    return create_blank_image()
