"""crypswolfy69 nodes — Alchemist, Prompt Rotate, and HF Model Downloader."""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .alchemist import routes as _alchemist_routes
from .alchemist.alchemist_node import (
    AlchemistDatasetCaptioner,
    AlchemistH3,
    AlchemistKrea2,
    AlchemistLoraCaption,
    AlchemistPrompt,
)
from .hf_downloader import live_browse as _hf_browse  # noqa: F401
from .hf_downloader import server as _hf_server  # noqa: F401
from .prompt_rotate.prompt_rotate import PromptRotate, PromptRotatePick

_alchemist_routes.register_routes()

WEB_DIRECTORY = "./web"

# Do not set NODE_CLASS_MAPPINGS — ComfyUI then treats this pack as V1
# and never calls comfy_entrypoint.


class Crypswolfy69Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            AlchemistKrea2,
            AlchemistLoraCaption,
            AlchemistH3,
            AlchemistDatasetCaptioner,
            AlchemistPrompt,
            PromptRotate,
            PromptRotatePick,
        ]


async def comfy_entrypoint() -> Crypswolfy69Extension:
    return Crypswolfy69Extension()


__all__ = [
    "comfy_entrypoint",
    "WEB_DIRECTORY",
    "AlchemistKrea2",
    "AlchemistLoraCaption",
    "AlchemistH3",
    "AlchemistDatasetCaptioner",
    "AlchemistPrompt",
    "PromptRotate",
    "PromptRotatePick",
]
