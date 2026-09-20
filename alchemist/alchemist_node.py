"""Alchemist nodes.

`AlchemistPrompt` does every engine in one node, which means a lot of widgets.
The three focused nodes below expose only what their job needs, so pick one of
those unless you genuinely want to switch engine on the fly.
"""

from pathlib import Path

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, ui

try:
    from .llm_client import REASONING_LEVELS, LLMError, chat, file_to_data_url, image_to_data_url
    from .prompts import (
        CAPTION_MODES,
        ENGINES,
        LORA_KINDS,
        build_system_prompt,
        build_user_message,
        normalize_caption_mode,
        normalize_engine,
    )
    from .providers import PROVIDER_ORDER, PROVIDERS, get_provider, resolve_key, save_key
    from . import routes
except ImportError:
    from llm_client import REASONING_LEVELS, LLMError, chat, file_to_data_url, image_to_data_url
    from prompts import (
        CAPTION_MODES,
        ENGINES,
        LORA_KINDS,
        build_system_prompt,
        build_user_message,
        normalize_caption_mode,
        normalize_engine,
    )
    from providers import PROVIDER_ORDER, PROVIDERS, get_provider, resolve_key, save_key
    import routes

routes.register_routes()

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_ALL_MODELS = sorted({m for p in PROVIDERS.values() for m in p.fallback_models})
DEFAULT_MODEL = PROVIDERS["xai"].fallback_models[0]
ACTIONS = ["idle", "VAULT KEY", "FETCH MODELS", "PREVIEW"]


def _provider_labels() -> list[str]:
    return [PROVIDERS[k].label for k in PROVIDER_ORDER]


def _provider_from_label(label: str) -> str:
    text = (label or "").strip()
    for key in PROVIDER_ORDER:
        if PROVIDERS[key].label == text or key == text.lower():
            return key
    return "xai"


# --------------------------------------------------------------------------
# Shared widget groups. Anything a user rarely touches is marked advanced so
# the frontend tucks it away instead of burying the three inputs that matter.
# --------------------------------------------------------------------------


def _action_input() -> io.Combo.Input:
    return io.Combo.Input(
        "action",
        options=ACTIONS,
        default="idle",
        tooltip="Runs in the browser and resets to idle. Never queues a generation.",
    )


def _provider_inputs(with_reasoning: bool = True) -> list:
    rows = [
        io.Combo.Input(
            "provider",
            options=_provider_labels(),
            default=PROVIDERS["xai"].label,
            tooltip="Which LLM answers. Use FETCH MODELS after switching.",
        ),
        io.Combo.Input(
            "model",
            options=_ALL_MODELS or [DEFAULT_MODEL],
            default=DEFAULT_MODEL,
            tooltip="Model id. FETCH MODELS repopulates this from the live API.",
        ),
    ]
    if with_reasoning:
        rows.append(
            io.Combo.Input(
                "reasoning",
                options=list(REASONING_LEVELS),
                default="provider default",
                tooltip=(
                    "Only bites on reasoning models, where it is the difference between "
                    "9s and 42s. Ignored elsewhere."
                ),
            )
        )
    return rows


def _tuning_inputs() -> list:
    """Advanced knobs. Correct defaults matter more than easy access."""
    return [
        io.String.Input(
            "extra_direction",
            multiline=True,
            default="",
            optional=True,
            advanced=True,
            placeholder="optional: 'more wet asphalt, keep the red coat'",
        ),
        io.String.Input(
            "api_key",
            default="",
            optional=True,
            advanced=True,
            placeholder="paste key, then run VAULT KEY",
            tooltip="Leave empty to use the vaulted key or the provider env var.",
        ),
        io.Float.Input(
            "temperature", default=0.8, min=0.0, max=2.0, step=0.05, optional=True, advanced=True
        ),
        io.Int.Input(
            "max_tokens", default=1400, min=64, max=32000, step=64, optional=True, advanced=True
        ),
        io.Boolean.Input(
            "remember_key",
            default=True,
            label_on="vault on run",
            label_off="do not store",
            optional=True,
            advanced=True,
        ),
        io.String.Input(
            "custom_base_url",
            default="",
            optional=True,
            advanced=True,
            placeholder="http://127.0.0.1:11434/v1",
            tooltip="Override the provider endpoint. Required for the Custom provider.",
        ),
        io.Int.Input(
            "reroll",
            default=0,
            min=0,
            max=0xFFFFFFFF,
            step=1,
            control_after_generate=True,
            optional=True,
            advanced=True,
            tooltip=(
                "Output is cached while inputs are unchanged, so repeat queues are free. "
                "Bump this to force a fresh call."
            ),
        ),
    ]


def _outputs() -> list:
    return [
        io.String.Output("prompt", display_name="prompt"),
        io.String.Output("reasoning", display_name="reasoning"),
        io.String.Output("model_used", display_name="model_used"),
        io.String.Output("status", display_name="status"),
    ]


def _run(
    *,
    engine: str,
    provider_label: str,
    model: str,
    seed: str,
    nsfw: bool,
    reasoning: str = "provider default",
    caption_mode: str = "regular",
    source_image=None,
    control_image=None,
    extra_direction: str = "",
    api_key: str = "",
    temperature: float = 0.8,
    max_tokens: int = 1400,
    remember_key: bool = True,
    custom_base_url: str = "",
    lora_kind: str = "character",
    trigger_word: str = "",
    caption_words: int = 45,
    omit_terms: str = "",
    controlnet_lock: str = "auto",
    edit_instruction: str = "",
    clip_seconds: float = 8.0,
    h3_aspect: str = "16:9",
    h3_music: str = "auto",
) -> io.NodeOutput:
    prov_key = _provider_from_label(provider_label)
    if remember_key and (api_key or "").strip():
        save_key(prov_key, api_key)

    engine = normalize_engine(engine)
    mode = normalize_caption_mode(caption_mode)

    src = image_to_data_url(source_image)
    ctl = image_to_data_url(control_image) if mode == "controlnet" else None
    images = [u for u in (src, ctl) if u]

    seed = (seed or "").strip()
    if not seed and not src and not (edit_instruction or "").strip():
        raise RuntimeError("Nothing to work from. Give a seed prompt or plug in a source image.")

    system = build_system_prompt(
        engine=engine,
        caption_mode=mode,
        nsfw=bool(nsfw),
        controlnet_lock=str(controlnet_lock),
        has_control_image=bool(ctl),
        lora_kind=str(lora_kind),
        trigger_word=str(trigger_word),
        caption_target_words=int(caption_words),
    )
    user = build_user_message(
        seed=seed,
        extra=extra_direction,
        engine=engine,
        caption_mode=mode,
        edit_instruction=edit_instruction,
        has_source_image=bool(src),
        has_control_image=bool(ctl),
        clip_seconds=float(clip_seconds),
        aspect=str(h3_aspect),
        music=str(h3_music),
        trigger_word=str(trigger_word),
        lora_kind=str(lora_kind),
        omit_extra=str(omit_terms),
    )

    try:
        text, reasoning_text, used = chat(
            provider=prov_key,
            model=model,
            system=system,
            user=user,
            images=images,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            inline_key=api_key,
            custom_base_url=custom_base_url,
            reasoning=str(reasoning),
        )
    except LLMError as err:
        raise RuntimeError(str(err)) from err

    if text.strip().startswith("REFUSED:"):
        raise RuntimeError(f"Model refused: {text.strip()}")

    vision = "IMG+CN" if (src and ctl) else "IMG" if src else "CN" if ctl else "TEXT"
    status = (
        f"{get_provider(prov_key).label} • {used} • {engine} • {vision} • "
        f"{len(text)} chars • {'NSFW' if nsfw else 'SFW'}"
    )
    return io.NodeOutput(text, reasoning_text, used, status, ui=ui.PreviewText(text))


class AlchemistKrea2(io.ComfyNode):
    """Krea 2 generation prompt. Six widgets; the rest is advanced."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AlchemistKrea2",
            display_name="✦ ALCHEMIST // KREA 2",
            category="prompting/alchemist",
            description=(
                "Expand a lazy caption into one natural-language Krea 2 prompt paragraph. "
                "Plug a source image in to describe a real photo instead."
            ),
            search_aliases=["krea2 prompt", "alchemist krea", "prompt expander", "magic prompt"],
            is_output_node=True,
            inputs=[
                _action_input(),
                *_provider_inputs(),
                io.String.Input(
                    "simple_prompt",
                    multiline=True,
                    default="a woman standing in neon rain",
                    placeholder="the lazy caption. The LLM does the baroque part.",
                ),
                io.Boolean.Input("nsfw", default=True, label_on="NSFW", label_off="SFW"),
                io.Image.Input("source_image", optional=True),
                io.Image.Input(
                    "control_image",
                    optional=True,
                    advanced=True,
                    tooltip="Only used when caption_mode is controlnet.",
                ),
                io.Combo.Input(
                    "caption_mode",
                    options=list(CAPTION_MODES),
                    default="regular",
                    optional=True,
                    advanced=True,
                ),
                io.Combo.Input(
                    "controlnet_lock",
                    options=["auto", "depth", "pose", "depth+pose", "canny", "lineart", "none"],
                    default="auto",
                    optional=True,
                    advanced=True,
                ),
                io.String.Input(
                    "edit_instruction", multiline=True, default="", optional=True, advanced=True
                ),
                *_tuning_inputs(),
            ],
            outputs=_outputs(),
        )

    @classmethod
    def execute(cls, action="idle", reroll=0, **kw) -> io.NodeOutput:
        del action, reroll
        return _run(
            engine="krea2",
            provider_label=kw.pop("provider"),
            model=kw.pop("model"),
            seed=kw.pop("simple_prompt", ""),
            nsfw=kw.pop("nsfw", True),
            **kw,
        )


class AlchemistLoraCaption(io.ComfyNode):
    """Krea 2 LoRA training caption. Not a prompt — nearly the inverse."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AlchemistLoraCaption",
            display_name="✦ ALCHEMIST // LORA CAPTION",
            category="prompting/alchemist",
            description=(
                "Write one training caption. Describes what should stay promptable and stays "
                "silent about whatever must bind to the trigger. lora_kind picks the rule."
            ),
            search_aliases=["lora caption", "training caption", "sidecar", "krea2 lora"],
            is_output_node=True,
            inputs=[
                _action_input(),
                *_provider_inputs(),
                io.Combo.Input(
                    "lora_kind",
                    options=list(LORA_KINDS),
                    default="body concept",
                    tooltip="Decides what is omitted so it binds to the trigger.",
                ),
                io.String.Input(
                    "trigger_word", default="", placeholder="e.g. mivra",
                    tooltip="Placed first in every caption.",
                ),
                io.Image.Input("source_image", optional=True),
                io.String.Input(
                    "simple_prompt",
                    multiline=True,
                    default="",
                    optional=True,
                    placeholder="what the image contains, if no image is plugged in",
                ),
                io.Boolean.Input("nsfw", default=True, label_on="NSFW", label_off="SFW"),
                io.Int.Input(
                    "caption_words", default=45, min=20, max=70, step=5, optional=True, advanced=True
                ),
                io.String.Input(
                    "omit_terms",
                    default="",
                    optional=True,
                    advanced=True,
                    placeholder="extra things to never mention",
                ),
                *_tuning_inputs(),
            ],
            outputs=_outputs(),
        )

    @classmethod
    def execute(cls, action="idle", reroll=0, **kw) -> io.NodeOutput:
        del action, reroll
        return _run(
            engine="krea2_lora",
            provider_label=kw.pop("provider"),
            model=kw.pop("model"),
            seed=kw.pop("simple_prompt", ""),
            nsfw=kw.pop("nsfw", True),
            **kw,
        )


class AlchemistH3(io.ComfyNode):
    """MiniMax H3 three-field video document."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AlchemistH3",
            display_name="✦ ALCHEMIST // MINIMAX H3",
            category="prompting/alchemist",
            description=(
                "Write the official H3 document: integrated_multimodal_description, "
                "overall_soundscape and non_diegetic_music."
            ),
            search_aliases=["minimax", "h3", "hailuo", "video prompt"],
            is_output_node=True,
            inputs=[
                _action_input(),
                *_provider_inputs(),
                io.String.Input(
                    "simple_prompt",
                    multiline=True,
                    default="a baker opens the shutters at dawn",
                    placeholder="the shot you want",
                ),
                io.Float.Input("clip_seconds", default=8.0, min=4.0, max=15.0, step=0.5),
                io.Combo.Input(
                    "h3_aspect", options=["16:9", "9:16", "1:1", "4:3", "21:9"], default="16:9"
                ),
                io.Boolean.Input("nsfw", default=True, label_on="NSFW", label_off="SFW"),
                io.Image.Input("source_image", optional=True),
                io.Combo.Input(
                    "h3_music",
                    options=["auto", "none", "tense strings", "warm piano", "synth pulse"],
                    default="auto",
                    optional=True,
                    advanced=True,
                ),
                *_tuning_inputs(),
            ],
            outputs=_outputs(),
        )

    @classmethod
    def execute(cls, action="idle", reroll=0, **kw) -> io.NodeOutput:
        del action, reroll
        return _run(
            engine="h3",
            provider_label=kw.pop("provider"),
            model=kw.pop("model"),
            seed=kw.pop("simple_prompt", ""),
            nsfw=kw.pop("nsfw", True),
            **kw,
        )


class AlchemistPrompt(io.ComfyNode):
    """All engines in one node. Kept for existing workflows."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AlchemistPrompt",
            display_name="✦ ALCHEMIST // ANY ENGINE",
            category="prompting/alchemist",
            description=(
                "Every engine behind one target_engine switch. If the widget count is "
                "getting in your way, use the focused KREA 2, LORA CAPTION or MINIMAX H3 "
                "nodes instead."
            ),
            search_aliases=["alchemist", "any llm", "prompt expander", "captioner"],
            is_output_node=True,
            inputs=[
                _action_input(),
                *_provider_inputs(),
                io.Combo.Input("target_engine", options=list(ENGINES), default="krea2"),
                io.String.Input(
                    "simple_prompt",
                    multiline=True,
                    default="a woman standing in neon rain",
                ),
                io.Boolean.Input("nsfw", default=True, label_on="NSFW", label_off="SFW"),
                io.Image.Input("source_image", optional=True),
                io.Image.Input("control_image", optional=True, advanced=True),
                io.Combo.Input(
                    "caption_mode", options=list(CAPTION_MODES), default="regular",
                    optional=True, advanced=True,
                ),
                io.Combo.Input(
                    "lora_kind", options=list(LORA_KINDS), default="character",
                    optional=True, advanced=True,
                ),
                io.String.Input("trigger_word", default="", optional=True, advanced=True),
                io.Int.Input(
                    "caption_words", default=45, min=20, max=70, step=5, optional=True, advanced=True
                ),
                io.String.Input("omit_terms", default="", optional=True, advanced=True),
                io.Combo.Input(
                    "controlnet_lock",
                    options=["auto", "depth", "pose", "depth+pose", "canny", "lineart", "none"],
                    default="auto",
                    optional=True,
                    advanced=True,
                ),
                io.String.Input(
                    "edit_instruction", multiline=True, default="", optional=True, advanced=True
                ),
                io.Float.Input(
                    "clip_seconds", default=8.0, min=4.0, max=15.0, step=0.5,
                    optional=True, advanced=True,
                ),
                io.Combo.Input(
                    "h3_aspect", options=["16:9", "9:16", "1:1", "4:3", "21:9"], default="16:9",
                    optional=True, advanced=True,
                ),
                io.Combo.Input(
                    "h3_music",
                    options=["auto", "none", "tense strings", "warm piano", "synth pulse"],
                    default="auto",
                    optional=True,
                    advanced=True,
                ),
                *_tuning_inputs(),
            ],
            outputs=_outputs(),
        )

    @classmethod
    def execute(cls, action="idle", reroll=0, **kw) -> io.NodeOutput:
        del action, reroll
        return _run(
            engine=kw.pop("target_engine", "krea2"),
            provider_label=kw.pop("provider"),
            model=kw.pop("model"),
            seed=kw.pop("simple_prompt", ""),
            nsfw=kw.pop("nsfw", True),
            **kw,
        )


class AlchemistDatasetCaptioner(io.ComfyNode):
    """Caption a whole folder into .txt sidecars for LoRA training."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AlchemistDatasetCaptioner",
            display_name="✦ ALCHEMIST // DATASET CAPTIONER",
            category="prompting/alchemist",
            description=(
                "Walk a folder and write a same-stem .txt caption beside each image, using the "
                "Krea 2 LoRA training recipe. Set overwrite off to only fill in the gaps."
            ),
            search_aliases=[
                "dataset captioner", "batch caption", "lora captions", "sidecar", "caption folder",
            ],
            is_output_node=True,
            not_idempotent=True,
            inputs=[
                _action_input(),
                io.String.Input(
                    "folder",
                    default="/workspace/Pawg TS",
                    tooltip="Folder of images. Captions land next to them as .txt.",
                ),
                *_provider_inputs(with_reasoning=False),
                io.Combo.Input(
                    "reasoning",
                    options=list(REASONING_LEVELS),
                    default="minimal",
                    tooltip="Paid per image here, so keep it low.",
                ),
                io.Combo.Input("lora_kind", options=list(LORA_KINDS), default="body concept"),
                io.String.Input("trigger_word", default="", placeholder="e.g. mivra"),
                io.Boolean.Input(
                    "dry_run", default=True, label_on="dry run", label_off="write files"
                ),
                io.Int.Input(
                    "limit", default=0, min=0, max=5000, step=1,
                    tooltip="0 = no cap. Set 3 for a cheap sanity check first.",
                ),
                io.Boolean.Input(
                    "overwrite", default=False, label_on="overwrite all", label_off="only missing"
                ),
                io.Boolean.Input("nsfw", default=True, label_on="NSFW", label_off="SFW"),
                io.Int.Input(
                    "caption_words", default=45, min=20, max=70, step=5, optional=True, advanced=True
                ),
                io.String.Input("omit_terms", default="", optional=True, advanced=True),
                io.String.Input(
                    "extra_direction", multiline=True, default="", optional=True, advanced=True
                ),
                io.String.Input("api_key", default="", optional=True, advanced=True),
                io.Float.Input(
                    "temperature", default=0.7, min=0.0, max=2.0, step=0.05,
                    optional=True, advanced=True,
                ),
                io.String.Input("custom_base_url", default="", optional=True, advanced=True),
            ],
            outputs=[
                io.String.Output("report", display_name="report"),
                io.Int.Output("captioned", display_name="captioned"),
                io.Int.Output("skipped", display_name="skipped"),
                io.Int.Output("failed", display_name="failed"),
            ],
        )

    @classmethod
    def execute(
        cls,
        folder,
        provider,
        model,
        action="idle",
        reasoning="minimal",
        lora_kind="body concept",
        trigger_word="",
        dry_run=True,
        limit=0,
        overwrite=False,
        nsfw=True,
        caption_words=45,
        omit_terms="",
        extra_direction="",
        api_key="",
        temperature=0.7,
        custom_base_url="",
    ) -> io.NodeOutput:
        del action
        root = Path(str(folder).strip())
        if not root.is_dir():
            raise RuntimeError(f"Not a folder: {root}")

        prov_key = _provider_from_label(provider)
        if (api_key or "").strip():
            save_key(prov_key, api_key)
        if get_provider(prov_key).needs_key and not resolve_key(prov_key, api_key):
            raise RuntimeError(f"No API key for {get_provider(prov_key).label}. Vault one first.")

        images = sorted(
            (f for f in root.iterdir() if f.suffix.lower() in IMAGE_EXTS), key=lambda f: f.name
        )
        if not images:
            raise RuntimeError(f"No images in {root}")

        todo = [f for f in images if overwrite or not f.with_suffix(".txt").exists()]
        skipped = len(images) - len(todo)
        if limit:
            todo = todo[: int(limit)]

        system = build_system_prompt(
            engine="krea2_lora",
            nsfw=bool(nsfw),
            lora_kind=str(lora_kind),
            trigger_word=str(trigger_word),
            caption_target_words=int(caption_words),
        )

        lines = [
            f"folder: {root}",
            f"images: {len(images)} | to caption: {len(todo)} | already had .txt: {skipped}",
            f"provider: {get_provider(prov_key).label} | model: {model} | kind: {lora_kind}",
            f"reasoning: {reasoning}",
            f"mode: {'DRY RUN (nothing written)' if dry_run else 'WRITING .txt sidecars'}",
            "",
        ]
        done = 0
        failed = 0

        for img in todo:
            data_url = file_to_data_url(str(img))
            if not data_url:
                lines.append(f"FAIL {img.name}: cannot read image")
                failed += 1
                continue
            user = build_user_message(
                engine="krea2_lora",
                has_source_image=True,
                extra=extra_direction,
                trigger_word=str(trigger_word),
                lora_kind=str(lora_kind),
                omit_extra=str(omit_terms),
            )
            try:
                text, _, _ = chat(
                    provider=prov_key,
                    model=model,
                    system=system,
                    user=user,
                    images=[data_url],
                    temperature=float(temperature),
                    max_tokens=400,
                    inline_key=api_key,
                    custom_base_url=custom_base_url,
                    reasoning=str(reasoning),
                )
            except LLMError as err:
                lines.append(f"FAIL {img.name}: {err}")
                failed += 1
                continue

            caption = " ".join(text.split())
            if caption.startswith("REFUSED:"):
                lines.append(f"REFUSED {img.name}: {caption}")
                failed += 1
                continue

            if not dry_run:
                img.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
            done += 1
            lines.append(f"{img.name} ({len(caption.split())}w): {caption}")

        lines.insert(5, f"result: {done} captioned, {skipped} skipped, {failed} failed\n")
        report = "\n".join(lines)
        return io.NodeOutput(report, done, skipped, failed, ui=ui.PreviewText(report))


class AlchemistExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            AlchemistKrea2,
            AlchemistLoraCaption,
            AlchemistH3,
            AlchemistDatasetCaptioner,
            AlchemistPrompt,
        ]


async def comfy_entrypoint() -> AlchemistExtension:
    return AlchemistExtension()
