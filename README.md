# ComfyUI-FLUX2MAX-KLEIN9B-API-Nodes

ComfyUI **V3** node pack for all **FLUX.2** models via the [BFL API](https://api.bfl.ai/).

> 🔀 Forked from [gelasdev/ComfyUI-FLUX-BFL-API](https://github.com/gelasdev/ComfyUI-FLUX-BFL-API). Rewritten as V3 async nodes with direct IMAGE inputs.

## Nodes

| Node | Display Name | Images | Key Features |
|---|---|---|---|
| **Flux2Max** | FLUX.2 [Max] (BFL) | 8 | Prompt upsampling, transparent BG |
| **Flux2Pro** | FLUX.2 [Pro] (BFL) | 8 | Prompt upsampling, transparent BG |
| **Flux2ProPreview** | FLUX.2 [Pro] Preview (BFL) | 8 | Prompt upsampling, transparent BG |
| **Flux2Klein9B** | FLUX.2 [Klein 9B] (BFL) | 4 | Transparent BG |
| **Flux2Klein4B** | FLUX.2 [Klein 4B] (BFL) | 4 | Transparent BG |
| **Flux2Klein9BKV** | FLUX.2 [Klein 9B KV] (BFL) | 4 | Transparent BG |
| **Flux2Flex** | FLUX.2 [Flex] (BFL) | 8 | Guidance, steps, prompt upsampling |
| **FluxConfig** | Flux Config (BFL) | — | API key + region config |

### All generation nodes

- ✅ Accept ComfyUI `IMAGE` tensors directly (no separate base64 node needed)
- ✅ Convert to base64 internally
- ✅ Async polling with progress bar
- ✅ Seed with `control_after_generate` (fixed / randomize / increment / decrement)
- ✅ Return blank image on failure instead of crashing

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/olliethomas1992/ComfyUI-FLUX2MAX-KLEIN9B-API-Nodes.git
pip install -r ComfyUI-FLUX2MAX-KLEIN9B-API-Nodes/requirements.txt
```

## Configuration

1. Get an API key from [api.bfl.ai](https://api.bfl.ai/)
2. Either:
   - Add a **Flux Config** node and enter your key, or
   - Create `config.ini` in the node pack directory:
     ```ini
     [API]
     key = your_bfl_api_key
     ```

## Requirements

- ComfyUI with V3 node support
- Python 3.10+
- `torch`, `aiohttp`

## Credits

Original work by [@gelasdev](https://github.com/gelasdev), [@pleberer](https://github.com/pleberer), and [@Duanyll](https://github.com/Duanyll).
