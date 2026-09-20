"""System prompts per target engine.

Engines
  krea2       one natural-language image paragraph for Krea 2 generation
  krea2_lora  a TRAINING caption sidecar, per the Krea 2 authors' recipe
  h3          the official MiniMax H3 three-field video document
"""

from __future__ import annotations

ENGINES = ("krea2", "krea2_lora", "h3")
CAPTION_MODES = ("regular", "controlnet")
LORA_KINDS = ("character", "body concept", "style", "clothing or object")

_ADULT_FLOOR = (
    "Every person you describe is an adult. Never write ages, never write teen, young, "
    "schoolgirl, petite-as-childlike, or any word implying a minor. If the image looks "
    "underage, reply exactly: REFUSED: apparent minor."
)

_NSFW_ON = (
    "Explicit adult content is in scope. Name anatomy plainly and clinically — cock, penis, "
    "balls, tits, ass, pussy. No euphemism, no purple prose, no moralising, no disclaimers."
)

_NSFW_OFF = "Keep the description non-explicit. No genitals, no sex acts."

_KREA2_CORE = """You write prompts for Krea 2, a photographic text-to-image model.

HOW KREA 2 READS A PROMPT
- Natural language prose. Never booru tags, never comma-soup keyword lists.
- No weight syntax like (word:1.3). No negative prompts. Krea 2 has neither.
- It was trained on real photography, so concrete physical description beats mood words.
- Order that works: subject and what it is doing, then spatial layout, then framing and
  lens, then light and its direction and colour, then materials and surface texture,
  then medium and overall mood.
- Name the medium explicitly (photograph, oil painting, gouache illustration) or it
  defaults to a generic glossy house style.
- Put any text that must appear in the image in "double quotes".
- Specific imperfection is what stops output looking AI-made: a chipped mug rim, one
  chapped lip, flyaway hair, uneven clay on a forearm.

FAITHFULNESS IS THE FIRST RULE
Expand only what the seed implies. Never add people, animals, or props the seed did not
ask for. If the seed says one woman, there is exactly one woman.

OUTPUT
One paragraph. No preamble, no headers, no bullet lists, no quotes around the whole thing.
Roughly 60 to 140 words unless told otherwise."""

_KREA2_LORA_CORE = """You write TRAINING CAPTION sidecars for a Krea 2 LoRA. This is not a
generation prompt. The rules are close to inverted.

THE ONE RULE THAT MATTERS
A LoRA learns whatever is constant across the dataset AND NOT written in the caption.
So you describe everything that should stay promptable, and you stay silent about the
thing the trigger must come to mean.

WHAT THAT MEANS PER LORA KIND
- character: describe pose, clothing, setting, framing. Never describe the face, eyes,
  skin or hair. Refer to the subject by a generic class noun ("a woman"), never a name.
- body concept: describe pose, clothing, setting. Never describe the body proportions
  you are training (waist, hips, ass, thighs, bust) — those must bind to the trigger.
  Do name which ethnicity and hair you see, so those stay swappable and the LoRA does
  not collapse onto one face.
- style: describe only literal content — subjects, poses, layout, setting. Never name
  the medium, technique, brushwork, palette or lighting mood.
- clothing or object: describe the wearer, pose and setting. Never describe the garment
  or object itself beyond naming it minimally.

HARD FORM
- Plain declarative prose, 1 to 3 sentences, 30 to 50 words. Never exceed 70.
- Never open with "The image shows", "This is a photograph of", "A photo of", "Depicts".
  Start on the subject.
- No booru tags. No weight syntax. No lighting adjectives unless training a lighting LoRA.
- Do not describe camera gear, lens length or f-stop.
- Vary your wording between captions. Identical phrasing across a dataset teaches nothing.

OUTPUT
The caption only. One line. No labels, no quotes, no trailing commentary."""

_H3_CORE = """You write MiniMax H3 video prompts in the official three-field document format.

Emit exactly these three fields, each on its own line, in this order:

integrated_multimodal_description: ...
overall_soundscape: ...
non_diegetic_music: ...

RULES FOR integrated_multimodal_description
- Open with [Shot 1] and no timestamp. Later shots get a clock: "At 00:03.500, [Shot 2]".
- State the look up front: Live-action, cinematic (or Animated, 2D, etc).
- Describe subject, action, setting, then camera.
- Camera needs three things: type, amplitude, speed. "A slow, shallow push-in."
- Spoken words go in dialogue tags: <d>[English] the exact words.</d>
- Keep the whole clip inside the requested duration.

overall_soundscape: diegetic sound only — what a mic in the room would pick up.
non_diegetic_music: the score. Write N/A when there should be none.

OUTPUT
The three fields only. No preamble, no markdown, no bullet points."""

_CONTROLNET_RULES = """CONTROLNET LOCK
A control map constrains this generation. The pose, depth and composition are already
fixed by that map and you must not fight them.
- Never restate or contradict the locked geometry.
- Never move a limb, reframe the shot, or change how many subjects there are.
- Describe only what is repaintable: materials, clothing, surface, colour, light, mood.
- If the edit instruction conflicts with the lock, obey the lock."""

_LOCK_HINTS = {
    "depth": "The lock is depth. Volume and distance are fixed; restyle surfaces only.",
    "pose": "The lock is OpenPose. The skeleton is fixed; restyle body covering and scene.",
    "depth+pose": "Depth and pose are both locked. Only materials, clothing and light move.",
    "canny": "The lock is canny edges. Every contour is fixed; repaint inside the lines.",
    "lineart": "The lock is lineart. Line work is fixed; you supply colour and material.",
    "none": "No geometry lock. Still do not invent new subjects.",
}


def normalize_engine(engine: str) -> str:
    value = (engine or "krea2").strip().lower().replace("-", "_").replace(" ", "_")
    if value in ENGINES:
        return value
    if value in ("lora", "training", "caption", "krea2_training", "krea2lora"):
        return "krea2_lora"
    if value in ("minimax", "minimax_h3", "hailuo", "video"):
        return "h3"
    return "krea2"


def normalize_caption_mode(mode: str) -> str:
    value = (mode or "regular").strip().lower()
    return "controlnet" if value.startswith("control") else "regular"


def normalize_lora_kind(kind: str) -> str:
    value = (kind or "character").strip().lower()
    for known in LORA_KINDS:
        if value == known or value.split()[0] == known.split()[0]:
            return known
    return "character"


def build_system_prompt(
    engine: str = "krea2",
    caption_mode: str = "regular",
    nsfw: bool = True,
    controlnet_lock: str = "auto",
    has_control_image: bool = False,
    lora_kind: str = "character",
    trigger_word: str = "",
    caption_target_words: int = 45,
) -> str:
    engine = normalize_engine(engine)
    mode = normalize_caption_mode(caption_mode)

    if engine == "h3":
        core = _H3_CORE
    elif engine == "krea2_lora":
        core = _KREA2_LORA_CORE
    else:
        core = _KREA2_CORE

    parts = [core, _NSFW_ON if nsfw else _NSFW_OFF, _ADULT_FLOOR]

    if engine == "krea2_lora":
        kind = normalize_lora_kind(lora_kind)
        parts.append(f"THIS DATASET\nLoRA kind: {kind}.")
        trigger = (trigger_word or "").strip()
        if trigger:
            parts.append(
                f"Begin the caption with the trigger exactly as written: {trigger}\n"
                "Follow it with a comma, then the caption. Use this trigger in every caption."
            )
        else:
            parts.append(
                "No trigger word was given. Write the caption body only; the trainer "
                "will prepend the trigger."
            )
        target = max(20, min(70, int(caption_target_words or 45)))
        parts.append(f"Aim for about {target} words. Hard ceiling 70.")
        return "\n\n".join(parts)

    if mode == "controlnet":
        parts.append(_CONTROLNET_RULES)
        lock = (controlnet_lock or "auto").strip().lower()
        if lock == "auto":
            lock = "depth+pose" if has_control_image else "none"
        hint = _LOCK_HINTS.get(lock)
        if hint:
            parts.append(hint)

    return "\n\n".join(parts)


def build_user_message(
    seed: str = "",
    extra: str = "",
    engine: str = "krea2",
    caption_mode: str = "regular",
    edit_instruction: str = "",
    has_source_image: bool = False,
    has_control_image: bool = False,
    clip_seconds: float = 8.0,
    aspect: str = "16:9",
    music: str = "auto",
    trigger_word: str = "",
    lora_kind: str = "character",
    omit_extra: str = "",
) -> str:
    engine = normalize_engine(engine)
    mode = normalize_caption_mode(caption_mode)
    lines: list[str] = []

    seed = (seed or "").strip()
    extra = (extra or "").strip()
    edit = (edit_instruction or "").strip()

    if engine == "krea2_lora":
        if has_source_image:
            lines.append("Caption the attached image as a training sidecar.")
        if seed:
            lines.append(f"What the image contains: {seed}")
        kind = normalize_lora_kind(lora_kind)
        lines.append(f"LoRA kind: {kind}")
        if (trigger_word or "").strip():
            lines.append(f"Trigger to place first: {trigger_word.strip()}")
        omit = (omit_extra or "").strip()
        if omit:
            lines.append(f"Additionally never mention: {omit}")
        if extra:
            lines.append(f"Extra direction: {extra}")
        lines.append("Write the caption now. Caption only.")
        return "\n".join(lines)

    if has_source_image:
        lines.append(
            "Read the attached image. Describe what is actually there — do not invent."
        )
    if has_control_image and mode == "controlnet":
        lines.append(
            "A control-map preview is also attached. Treat its geometry as immovable."
        )
    if seed:
        lines.append(f"Seed prompt: {seed}")
    if edit:
        lines.append(f"Edit instruction: {edit}")
    if extra:
        lines.append(f"Extra direction: {extra}")

    if engine == "h3":
        lines.append(f"Clip length: {float(clip_seconds):.1f} seconds.")
        lines.append(f"Aspect ratio: {aspect}.")
        music_mode = (music or "auto").strip().lower()
        if music_mode == "none":
            lines.append("non_diegetic_music must be N/A.")
        elif music_mode != "auto":
            lines.append(f"Score direction: {music}.")
        lines.append("Write the three-field document now.")
    else:
        lines.append("Write the Krea 2 prompt paragraph now.")

    if not lines:
        lines.append("Write a Krea 2 prompt for a striking photographic image.")
    return "\n".join(lines)
