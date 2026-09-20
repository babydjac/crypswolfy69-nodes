from pathlib import Path
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, ui

try:
    from .parse_prompts import parse_prompts, pick_prompt
except ImportError:
    from parse_prompts import parse_prompts, pick_prompt

_DEFAULT_PATH = Path(__file__).resolve().parent / "defaults" / "pawg_ts_40.txt"
try:
    DEFAULT_PROMPTS = _DEFAULT_PATH.read_text(encoding="utf-8")
except OSError:
    DEFAULT_PROMPTS = ""


class PromptRotate(io.ComfyNode):
    """Paste many prompts. Outputs a STRING list (9-dot socket) so one Queue runs every prompt."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PromptRotate",
            display_name="✦ PROMPT ROTATE",
            category="prompting/queue",
            description=(
                "Paste a prompt dump. `prompt` is a STRING list (9 dots). "
                "Wire it into Krea 2 text encode and hit Queue once — Comfy walks the list."
            ),
            search_aliases=[
                "prompt list",
                "prompt rotate",
                "queue all",
                "batch prompts",
                "krea2",
                "prompt dump",
            ],
            is_output_node=True,
            not_idempotent=True,
            inputs=[
                io.Combo.Input(
                    "run",
                    options=["idle", "QUEUE ALL", "COUNT"],
                    default="idle",
                    tooltip="Set QUEUE ALL to enqueue one job per pasted prompt. COUNT just reports how many it parsed.",
                ),
                io.String.Input(
                    "prompts",
                    multiline=True,
                    default=DEFAULT_PROMPTS,
                    placeholder="Paste prompts here. Numbered 1. 2. 3. or blank-line separated.",
                    tooltip="The full dump. Numbered lists, blank-line blocks, or one prompt per line.",
                ),
                io.Int.Input(
                    "prompt_index",
                    default=1,
                    min=1,
                    max=9999,
                    step=1,
                    control_after_generate=True,
                    tooltip="1-based. Only used by QUEUE ALL / single-pick mode.",
                ),
                io.Combo.Input(
                    "split_mode",
                    options=["auto", "numbered", "blank lines", "one per line"],
                    default="auto",
                    tooltip="auto reads numbered lists first, then blank lines, then long single lines.",
                ),
                io.Boolean.Input(
                    "wrap",
                    default=True,
                    label_on="wrap",
                    label_off="clamp",
                    tooltip="If index is past the end, wrap around. Off clamps to the last prompt.",
                ),
                io.Boolean.Input(
                    "bump_graph_seeds",
                    default=True,
                    label_on="bump seeds",
                    label_off="same seed",
                    tooltip="QUEUE ALL adds +i to every seed / noise_seed in the graph so jobs do not clone.",
                ),
                io.Int.Input(
                    "start_seed",
                    default=1000,
                    min=0,
                    max=0xFFFFFFFF,
                    step=1,
                    tooltip="Seed list is start_seed, start_seed+1, ... Wire this to the sampler.",
                ),
                io.String.Input(
                    "file_prefix",
                    default="pawgts_",
                    tooltip="filename output is prefix + 01, 02, ...",
                ),
                io.String.Input(
                    "forced_prompt",
                    default="",
                    optional=True,
                    tooltip="Filled by QUEUE ALL per job. Leave empty.",
                ),
            ],
            outputs=[
                io.String.Output("prompt", display_name="prompt", is_output_list=True),
                io.Int.Output("index", display_name="index", is_output_list=True),
                io.Int.Output("total", display_name="total"),
                io.Int.Output("seed", display_name="seed", is_output_list=True),
                io.String.Output("filename", display_name="filename", is_output_list=True),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompts,
        prompt_index=1,
        split_mode="auto",
        wrap=True,
        bump_graph_seeds=True,
        start_seed=1000,
        file_prefix="pawgts_",
        forced_prompt="",
        run="idle",
    ) -> io.NodeOutput:
        del bump_graph_seeds, run, prompt_index, wrap, forced_prompt
        items = parse_prompts(prompts, split_mode)
        prefix = str(file_prefix or "pawgts_")
        indexes = list(range(1, len(items) + 1))
        seeds = [int(start_seed) + i for i in range(len(items))]
        filenames = [f"{prefix}{i:02d}" for i in indexes]
        preview = f"{len(items)} prompts"
        if items:
            preview += "\n" + items[0][:240]
        return io.NodeOutput(
            items,
            indexes,
            len(items),
            seeds,
            filenames,
            ui=ui.PreviewText(preview),
        )


class PromptRotatePick(io.ComfyNode):
    """Single prompt out of the dump, by index. Use when you want one job at a time."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PromptRotatePick",
            display_name="✦ PROMPT ROTATE // PICK ONE",
            category="prompting/queue",
            description="Pick one prompt from a dump by index. Single socket, not a list.",
            search_aliases=["prompt rotate pick", "prompt index", "one prompt"],
            not_idempotent=True,
            inputs=[
                io.String.Input(
                    "prompts",
                    multiline=True,
                    default=DEFAULT_PROMPTS,
                    tooltip="Same dump format as Prompt Rotate.",
                ),
                io.Int.Input(
                    "prompt_index",
                    default=1,
                    min=1,
                    max=9999,
                    step=1,
                    control_after_generate=True,
                    tooltip="1-based index. Set control_after_generate to increment for sequential runs.",
                ),
                io.Combo.Input(
                    "split_mode",
                    options=["auto", "numbered", "blank lines", "one per line"],
                    default="auto",
                ),
                io.Boolean.Input("wrap", default=True, label_on="wrap", label_off="clamp"),
                io.Int.Input("start_seed", default=1000, min=0, max=0xFFFFFFFF, step=1),
                io.String.Input("file_prefix", default="pawgts_"),
            ],
            outputs=[
                io.String.Output("prompt", display_name="prompt"),
                io.Int.Output("index", display_name="index"),
                io.Int.Output("total", display_name="total"),
                io.Int.Output("seed", display_name="seed"),
                io.String.Output("filename", display_name="filename"),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompts,
        prompt_index=1,
        split_mode="auto",
        wrap=True,
        start_seed=1000,
        file_prefix="pawgts_",
    ) -> io.NodeOutput:
        text, idx, total = pick_prompt(
            prompts, prompt_index, split_mode=split_mode, wrap=bool(wrap)
        )
        prefix = str(file_prefix or "pawgts_")
        return io.NodeOutput(
            text,
            idx,
            total,
            int(start_seed) + max(0, idx - 1),
            f"{prefix}{idx:02d}" if total else prefix,
            ui=ui.PreviewText(f"{idx}/{total}\n{text}"),
        )


class PromptRotateExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PromptRotate, PromptRotatePick]


async def comfy_entrypoint() -> PromptRotateExtension:
    return PromptRotateExtension()
