# crypswolfy69 nodes

One ComfyUI pack with three tools:

| Piece | What it does |
|---|---|
| **Alchemist** | Prompt + LoRA-caption nodes for Krea 2 and MiniMax H3, any LLM provider |
| **Prompt Rotate** | Paste a prompt list → STRING list socket, or queue one job per prompt |
| **HF Model Downloader** | Browse Hugging Face and download into the right `models/` folder |

Sources bundled from:

- https://github.com/babydjac/ComfyUI-Alchemist
- https://github.com/babydjac/ComfyUI-PromptRotate
- https://github.com/babydjac/ComfyUI_HF_ModelDownloader

## Install

**ComfyUI-Manager / Registry**

Search `crypswolfy69 nodes` and install.

**Manual**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/babydjac/crypswolfy69-nodes.git
pip install -r crypswolfy69-nodes/requirements.txt
```

Restart ComfyUI and hard-refresh the browser.

HF downloads also want `aria2c` on PATH.

## Nodes

- `AlchemistKrea2` / `AlchemistLoraCaption` / `AlchemistH3` / `AlchemistPrompt` / `AlchemistDatasetCaptioner`
- `PromptRotate` / `PromptRotatePick`
- HF Model Downloader UI (floating **Model Browser** / sidebar **HF Models**)

Existing workflows that used the standalone packs keep the same class names.

## License

MIT. See `LICENSE`.
