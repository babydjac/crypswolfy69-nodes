from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import secrets
import socket
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote, urlencode
from urllib.request import Request as UrlRequest
from urllib.request import urlopen as sync_urlopen

import aiohttp
import folder_paths
from aiohttp import web
from server import PromptServer

HF_API = "https://huggingface.co/api"
HF_WEB = "https://huggingface.co"

THIS_DIR = Path(__file__).resolve().parent
CACHE_DIR = THIS_DIR / ".cache"
CACHE_PATH = CACHE_DIR / "index.json"
CACHE_MAX_AGE_SECONDS = 6 * 60 * 60
CACHE_SCHEMA_VERSION = 4

DEFAULT_OWNERS = [
    "Comfy-Org",
    "Kijai",
    "black-forest-labs",
    "Tencent-Hunyuan",
    "Qwen",
    "lllyasviel",
    # Upscalers. These publish plain ESRGAN .pth files with no ComfyUI tags and
    # zero reported downloads, so they need the relaxed checks below to survive.
    "uwg",
    "Kim2091",
    "gemasai",
    "Phips",
]
DEFAULT_LIMIT_PER_OWNER = 30
STRICT_FILTER_DEFAULT = True
HF_MAX_REPO_TREE_CONCURRENCY = 6

MODEL_EXTENSIONS = {
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".gguf",
}

MIN_REPO_DOWNLOADS = 500
MIN_REPO_LIKES = 20

# Tags that mark a super-resolution repo. The classic ESRGAN publishers carry
# these instead of the ComfyUI/diffusers tags the main filter looks for.
UPSCALER_TAGS = {"super-resolution", "upscalers", "image-to-image"}

# Well known upscaler architectures and model families, plus the "4x-" style
# scale prefix. Needed because a name like 4x-UltraSharp.pth contains neither
# "upscale" nor "esrgan" and would otherwise be filed as a checkpoint.
UPSCALER_NAME_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:\d+x|x\d+)(?:[^a-z0-9]|$)"
    r"|esrgan|ultrasharp|ultramix|siax|nmkd|remacri|lollypop|foolhardy"
    r"|swinir|hat[-_]|realsr|bsrgan|dat[-_]|span[-_]|omnisr|nomos|ldsr"
    r"|superscale|super[-_]?resolution|upscal|skindiff|anime6b|lanczos",
    re.IGNORECASE,
)


def looks_like_upscaler(repo: Dict[str, object], path: str = "") -> bool:
    tags = {str(tag).lower() for tag in repo.get("tags", [])}
    if tags & UPSCALER_TAGS:
        return True
    repo_id = str(repo.get("id", ""))
    return bool(UPSCALER_NAME_RE.search(f"{repo_id} {path}"))

SHARD_FILE_RE = re.compile(r"\d{4,5}[-_]of[-_]\d{4,5}", re.IGNORECASE)
TRAINING_ARTIFACT_RE = re.compile(
    r"(optimizer|trainer[_-]?state|training[_-]?args|random[_-]?states?|global[_-]?step|ema[_-]?state)",
    re.IGNORECASE,
)
NOISE_NAME_RE = re.compile(r"(test|sample|example|demo)[-_]?(weights|model)?", re.IGNORECASE)
DIFFUSION_COMPONENT_RE = re.compile(
    r"(diffusion_pytorch_model|traj[_-]?extractor|feature[_-]?extractor|tokenizer|scheduler|safety[_-]?checker)",
    re.IGNORECASE,
)

MIN_SIZE_BY_CATEGORY = {
    "diffusion_models": 512 * 1024 * 1024,
    "checkpoints": 180 * 1024 * 1024,
    "unet": 256 * 1024 * 1024,
    "controlnet": 80 * 1024 * 1024,
    "text_encoders": 20 * 1024 * 1024,
    "clip": 20 * 1024 * 1024,
    "clip_vision": 20 * 1024 * 1024,
    "vae": 15 * 1024 * 1024,
    "upscale_models": 15 * 1024 * 1024,
    "latent_upscale_models": 15 * 1024 * 1024,
    "loras": 1 * 1024 * 1024,
    "model_patches": 1 * 1024 * 1024,
    "embeddings": 64 * 1024,
}

TITLE_STRIP_RE = re.compile(
    r"\b(fp16|fp32|bf16|int8|int4|ema|nonema|weights?|model)\b",
    re.IGNORECASE,
)

ALIASES = {
    "comfy_org": "Comfy-Org",
    "comfy-org": "Comfy-Org",
    "kijai": "Kijai",
    "black-forest-labs": "black-forest-labs",
    "qwen": "Qwen",
    "tencent-hunyuan": "Tencent-Hunyuan",
}

CATEGORY_ORDER = [
    "diffusion_models",
    "checkpoints",
    "model_patches",
    "text_encoders",
    "clip",
    "clip_vision",
    "vae",
    "loras",
    "controlnet",
    "upscale_models",
    "latent_upscale_models",
    "audio_encoders",
    "unet",
    "embeddings",
    "style_models",
    "hypernetworks",
    "photomaker",
    "gligen",
    "vae_approx",
]

KNOWN_CATEGORY_PARTS = {
    "audio_encoders": "audio_encoders",
    "checkpoints": "checkpoints",
    "clip": "clip",
    "clip_vision": "clip_vision",
    "controlnet": "controlnet",
    "diffusion_models": "diffusion_models",
    "embeddings": "embeddings",
    "gligen": "gligen",
    "hypernetworks": "hypernetworks",
    "latent_upscale_models": "latent_upscale_models",
    "loras": "loras",
    "model_patches": "model_patches",
    "photomaker": "photomaker",
    "style_models": "style_models",
    "text_encoders": "text_encoders",
    "unet": "unet",
    "upscale_models": "upscale_models",
    "vae": "vae",
    "vae_approx": "vae_approx",
}

# A client-supplied category is only ever used as a folder name under models/, so it has
# to be one we already know about.
VALID_CATEGORIES = frozenset(KNOWN_CATEGORY_PARTS.values())

TOKEN_CANDIDATE_PATHS = [
    THIS_DIR / ".hf_token",
    Path("/workspace/mod/.hf_token"),
]

FAMILY_KEYWORDS = {
    "WAN": ("wan2.2", "wan2.1", "wan2", "wan "),
    "FLUX": ("flux",),
    "QWEN": ("qwen",),
    "HUNYUAN": ("hunyuan",),
    "Z_IMAGE": ("z-image", "z_image", "zimage"),
    "TURBO": ("turbo",),
    "SDXL": ("sdxl", "stable-diffusion-xl"),
    "SD3": ("sd3", "stable-diffusion-3"),
    "SD15": ("sd1.5", "sd15", "stable-diffusion-v1", "stable-diffusion-1.5"),
    "PIXART": ("pixart",),
    "LTXV": ("ltxv", "ltx-video", "ltx"),
    "COGVIDEOX": ("cogvideox", "cogvideo"),
}

FAMILY_STOPWORDS = {
    "model",
    "models",
    "weights",
    "checkpoint",
    "checkpoints",
    "fp16",
    "bf16",
    "int8",
    "dev",
    "main",
    "latest",
    "diffusion",
    "comfy",
    "comfyui",
}

JOBS: Dict[str, Dict[str, object]] = {}
JOBS_LOCK = threading.Lock()
JOB_PROCESSES: Dict[str, subprocess.Popen] = {}
INDEX_BUILD_LOCK = threading.Lock()
MAX_STORED_JOBS = 120
TERMINAL_JOB_STATUSES = frozenset({"done", "error", "cancelled"})


def _repo_tree_revision(repo: Dict[str, object]) -> str:
    sha = repo.get("sha")
    if isinstance(sha, str):
        cleaned = sha.strip()
        if len(cleaned) >= 7:
            return cleaned
    return "main"


def _prune_finished_jobs() -> None:
    with JOBS_LOCK:
        if len(JOBS) <= MAX_STORED_JOBS:
            return
        finished: List[Tuple[str, int]] = []
        for jid, job in JOBS.items():
            status = str(job.get("status", "")).lower()
            if status in TERMINAL_JOB_STATUSES:
                finished.append((jid, int(job.get("created_at", 0) or 0)))
        finished.sort(key=lambda pair: pair[1])
        for jid, _ in finished:
            if len(JOBS) <= MAX_STORED_JOBS:
                break
            job = JOBS.get(jid)
            if job is None:
                continue
            if str(job.get("status", "")).lower() in TERMINAL_JOB_STATUSES:
                JOBS.pop(jid, None)
                JOB_PROCESSES.pop(jid, None)


def normalize_owner(name: str) -> str:
    return ALIASES.get(name.strip().lower(), name.strip())


def normalize_owner_list(names: Sequence[str]) -> List[str]:
    owners: List[str] = []
    seen: set[str] = set()
    for raw_name in names:
        owner = normalize_owner(str(raw_name)).strip()
        if not owner:
            continue
        key = owner.lower()
        if key in seen:
            continue
        seen.add(key)
        owners.append(owner)
    return owners


def parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_int(value: Optional[str], default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, parsed))


def _read_token_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def _token_with_source() -> Tuple[Optional[str], str]:
    for path in TOKEN_CANDIDATE_PATHS:
        token = _read_token_file(path)
        if token:
            return token, f"file:{path}"
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip(), "env"
    return None, "none"


def get_hf_token() -> Optional[str]:
    token, _source = _token_with_source()
    return token


def get_headers() -> Dict[str, str]:
    headers = {"User-Agent": "comfyui-hf-model-downloader/1.0"}
    token = get_hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def mask_token(token: Optional[str]) -> str:
    if not token:
        return ""
    clean = token.strip()
    if len(clean) <= 8:
        return "*" * len(clean)
    return f"{clean[:4]}...{clean[-4:]}"


async def request_json(
    session: aiohttp.ClientSession,
    url: str,
    retries: int = 3,
    timeout: int = 60,
) -> object:
    headers = get_headers()
    use_auth = "Authorization" in headers
    last_error: Optional[str] = None
    for attempt in range(1, retries + 1):
        try:
            req_headers = dict(headers)
            if not use_auth:
                req_headers.pop("Authorization", None)
            request_timeout = aiohttp.ClientTimeout(total=max(5, int(timeout)))
            async with session.get(url, headers=req_headers, timeout=request_timeout) as response:
                if response.status == 401 and use_auth:
                    # If token is stale, retry once without auth so public index still works.
                    await response.read()
                    use_auth = False
                    continue
                if response.status in {429, 500, 502, 503, 504} and attempt < retries:
                    await response.read()
                    await asyncio.sleep(attempt)
                    continue
                if response.status >= 400:
                    body = (await response.text())[:240]
                    raise RuntimeError(f"HTTP {response.status} for {url}: {body}")
                text = await response.text()
                return json.loads(text)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if isinstance(exc, json.JSONDecodeError):
                raise RuntimeError(f"Invalid JSON returned for {url}: {exc}") from exc
            if attempt == retries:
                break
            await asyncio.sleep(attempt)
        except RuntimeError as exc:
            last_error = str(exc)
            if attempt == retries:
                break
            await asyncio.sleep(attempt)
    raise RuntimeError(f"Failed request for {url}: {last_error}")


def relabel_nonstandard_dirs(parts: Sequence[str]) -> List[str]:
    remapped: List[str] = []
    for part in parts:
        key = part.lower()
        if key in {"upscalers", "upscaler", "upscale"}:
            remapped.append("upscale_models")
        elif key in {"lora", "lora_models"}:
            remapped.append("loras")
        elif key in {"text_encoder", "text_encoder_2", "text-encoder", "encoder", "encoders"}:
            remapped.append("text_encoders")
        elif key in {"transformer"}:
            remapped.append("diffusion_models")
        else:
            remapped.append(key)
    return remapped


def categorize_file(repo: Dict[str, object], path: str) -> str:
    lower_path = path.lower()
    parts = relabel_nonstandard_dirs(Path(path).parts[:-1])
    for part in parts:
        if part in KNOWN_CATEGORY_PARTS:
            return KNOWN_CATEGORY_PARTS[part]

    filename = Path(path).name.lower()
    tags = {str(tag).lower() for tag in repo.get("tags", [])}
    repo_name = str(repo.get("id", "")).split("/")[-1].lower()

    if "controlnet" in tags or "controlnet" in lower_path or repo_name.startswith("control_"):
        return "controlnet"
    if "lora" in filename or "lora" in lower_path:
        return "loras"
    if "upscal" in filename or "esrgan" in lower_path or looks_like_upscaler(repo, path):
        return "upscale_models"
    if "text_encoder" in lower_path or "text projection" in filename.replace("_", " "):
        return "text_encoders"
    if "clip_vision" in lower_path:
        return "clip_vision"
    if re.search(r"(^|[_-])clip([_-]|$)", filename) and "text_encoder" not in lower_path:
        return "clip"
    if "vae_approx" in lower_path:
        return "vae_approx"
    if "vae" in lower_path or filename.startswith("ae"):
        return "vae"
    if "ip_adapter" in lower_path or "patch" in lower_path:
        return "model_patches"
    if "embedding" in lower_path:
        return "embeddings"
    if "hypernetwork" in lower_path:
        return "hypernetworks"
    if "photomaker" in lower_path:
        return "photomaker"
    if "gligen" in lower_path:
        return "gligen"
    if "audio" in lower_path or "wav2vec" in lower_path:
        return "audio_encoders"
    if "unet" in lower_path:
        return "unet"
    if "diffusion-single-file" in tags or "comfyui" in tags:
        return "checkpoints"
    if "diffusers" in tags:
        return "diffusion_models"
    return "checkpoints"


def should_keep_repo(repo: Dict[str, object]) -> bool:
    tags = {str(tag).lower() for tag in repo.get("tags", [])}
    repo_id = str(repo.get("id", "")).lower()
    downloads = int(repo.get("downloads", 0) or 0)
    likes = int(repo.get("likes", 0) or 0)
    popular = downloads >= MIN_REPO_DOWNLOADS or likes >= MIN_REPO_LIKES
    if not popular:
        return False
    # Upscaler repos never carry the ComfyUI/diffusers tags, so accept them on
    # the strength of a super-resolution tag or a recognisable model name.
    if looks_like_upscaler(repo):
        return True
    return any(
        flag in tags
        for flag in {"comfyui", "diffusers", "diffusion-single-file", "controlnet", "gguf", "safetensors"}
    ) or "control" in repo_id or "flux" in repo_id or "wan" in repo_id or "comfy" in repo_id


async def fetch_top_repos(
    session: aiohttp.ClientSession,
    owner: str,
    limit: int,
) -> List[Dict[str, object]]:
    params = urlencode(
        {
            "author": owner,
            "limit": limit,
            "sort": "downloads",
            "direction": "-1",
            "full": "true",
        }
    )
    url = f"{HF_API}/models?{params}"
    payload = await request_json(session, url)
    if not isinstance(payload, list):
        return []
    repos = []
    for repo in payload:
        if not isinstance(repo, dict):
            continue
        if should_keep_repo(repo):
            repos.append(repo)
    return repos


async def fetch_repo_tree(
    session: aiohttp.ClientSession,
    repo_id: str,
    revision: str = "main",
) -> List[Dict[str, object]]:
    quoted_repo = quote(repo_id, safe="/")
    rev = (revision or "main").strip() or "main"
    quoted_rev = quote(rev, safe="")
    url = f"{HF_API}/models/{quoted_repo}/tree/{quoted_rev}?recursive=1&expand=false"
    payload = await request_json(session, url, timeout=120)
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


def is_shard_file(path: str) -> bool:
    filename = Path(path).name.lower()
    return bool(SHARD_FILE_RE.search(filename))


def is_training_artifact(path: str) -> bool:
    lowered = path.lower()
    if TRAINING_ARTIFACT_RE.search(lowered):
        return True
    filename = Path(lowered).name
    if filename.endswith(".index.json") or filename.endswith(".md"):
        return True
    return False


def is_noise_name(path: str) -> bool:
    filename = Path(path).stem.lower()
    return bool(NOISE_NAME_RE.search(filename))


def is_diffusion_component_artifact(path: str, category: str) -> bool:
    if category != "diffusion_models":
        return False
    lowered = path.lower()
    # Keep only proper model files, not diffusers internals/components.
    return bool(DIFFUSION_COMPONENT_RE.search(lowered))


def minimum_size_for_category(category: str) -> int:
    return MIN_SIZE_BY_CATEGORY.get(category, 1 * 1024 * 1024)


def is_size_too_small(category: str, size: object) -> bool:
    if not isinstance(size, int) or size <= 0:
        return False
    return size < minimum_size_for_category(category)


def should_keep_model_file(
    repo: Dict[str, object],
    path: str,
    category: str,
    size: object,
    strict_filter: bool,
) -> bool:
    suffix = Path(path).suffix.lower()
    if suffix not in MODEL_EXTENSIONS:
        return False
    if strict_filter:
        if is_shard_file(path):
            return False
        if is_diffusion_component_artifact(path, category):
            return False
        if is_training_artifact(path):
            return False
        if is_noise_name(path):
            return False
        if is_size_too_small(category, size):
            return False
    # Keep GGUF focused on known model families to avoid random tiny tool models.
    if strict_filter and suffix == ".gguf":
        lowered = path.lower()
        if not any(keyword in lowered for keyword in ("flux", "wan", "qwen", "hunyuan", "cogvideo", "ltx")):
            return False
    return True


def prettify_title(raw: str) -> str:
    title = raw.replace("_", " ").replace("-", " ").strip()
    title = re.sub(r"\s+", " ", title).strip()
    return title


def title_from_path(path: str, repo_id: str) -> str:
    stem = Path(path).stem
    if stem in {"model", "diffusion_pytorch_model", "pytorch_model", "ae"}:
        repo_part = repo_id.split("/")[-1]
        parent = Path(path).parent.name
        if parent and parent not in {".", ""} and parent not in {"text_encoder", "transformer", "vae"}:
            return prettify_title(f"{repo_part} {parent}")
        return prettify_title(repo_part)
    return prettify_title(stem)


def _clean_tokens(raw: str) -> List[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9]+", raw.lower()) if token]


def detect_family(repo_id: str, path: str) -> str:
    blob = f"{repo_id}/{path}".lower()
    for family, keywords in FAMILY_KEYWORDS.items():
        if any(keyword in blob for keyword in keywords):
            return family

    for candidate in (Path(path).stem, repo_id.split("/")[-1]):
        for token in _clean_tokens(candidate):
            if token.isdigit():
                continue
            if token in FAMILY_STOPWORDS:
                continue
            if len(token) < 3:
                continue
            return token.upper()
    return "MISC"


def _normalized_title_key(title: str) -> str:
    lowered = title.lower()
    lowered = TITLE_STRIP_RE.sub(" ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered or title.lower()


def _quality_score(item: Dict[str, object]) -> int:
    suffix = Path(str(item.get("filename", ""))).suffix.lower()
    extension_bonus = {
        ".safetensors": 600,
        ".ckpt": 420,
        ".pt": 260,
        ".pth": 240,
        ".gguf": 180,
    }.get(suffix, 0)
    size = int(item.get("size", 0) or 0)
    downloads = int(item.get("downloads", 0) or 0)
    likes = int(item.get("likes", 0) or 0)
    filename = str(item.get("filename", "")).lower()
    penalty = 0
    if filename.startswith(("model", "weights", "checkpoint")):
        penalty -= 40
    return extension_bonus + min(size // (1024 * 1024), 500) + (downloads // 300) + (likes * 2) + penalty


def dedupe_items(items: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    picked: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    for item in items:
        repo_id = str(item.get("repo_id", ""))
        category = str(item.get("category", "checkpoints"))
        family = str(item.get("family", "MISC"))
        title_key = _normalized_title_key(str(item.get("title", "")))
        key = (repo_id, category, family, title_key)
        previous = picked.get(key)
        if previous is None or _quality_score(item) > _quality_score(previous):
            picked[key] = item
    return list(picked.values())


async def build_index(owners: Sequence[str], limit_per_owner: int, strict_filter: bool) -> Dict[str, object]:
    items: List[Dict[str, object]] = []
    errors: List[str] = []
    seen: set[Tuple[str, str]] = set()
    timeout = aiohttp.ClientTimeout(total=180, connect=25, sock_connect=25, sock_read=120)
    connector = aiohttp.TCPConnector(limit=max(8, HF_MAX_REPO_TREE_CONCURRENCY * 2))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        repos_by_id: Dict[str, Dict[str, object]] = {}
        for owner in owners:
            try:
                repos = await fetch_top_repos(session, owner, limit_per_owner)
            except Exception as exc:
                errors.append(f"{owner}: {exc}")
                continue

            for repo in repos:
                repo_id = str(repo.get("id", ""))
                if "/" not in repo_id:
                    continue
                repos_by_id[repo_id] = repo

        repo_tree_semaphore = asyncio.Semaphore(HF_MAX_REPO_TREE_CONCURRENCY)

        async def fetch_repo_tree_job(repo_id: str) -> Tuple[str, List[Dict[str, object]], Optional[str]]:
            repo_meta = repos_by_id.get(repo_id, {})
            revision = _repo_tree_revision(repo_meta)
            async with repo_tree_semaphore:
                try:
                    tree = await fetch_repo_tree(session, repo_id, revision)
                except Exception as exc:
                    return repo_id, [], str(exc)
            return repo_id, tree, None

        files_scanned = 0
        tasks = [asyncio.create_task(fetch_repo_tree_job(repo_id)) for repo_id in repos_by_id]
        for task in asyncio.as_completed(tasks):
            repo_id, tree, err = await task
            if err:
                errors.append(f"{repo_id}: {err}")
                continue
            repo = repos_by_id.get(repo_id)
            if not repo:
                continue
            repo_revision = _repo_tree_revision(repo)

            for entry in tree:
                if entry.get("type") != "file":
                    continue
                path = str(entry.get("path", ""))
                category = categorize_file(repo, path)
                if not should_keep_model_file(repo, path, category, entry.get("size"), strict_filter):
                    continue
                key = (repo_id, path)
                if key in seen:
                    continue
                seen.add(key)
                family = detect_family(repo_id, path)
                title = title_from_path(path, repo_id)
                item = {
                    "id": f"{repo_id}:{path}",
                    "repo_id": repo_id,
                    "repo_revision": repo_revision,
                    "owner": repo_id.split("/")[0],
                    "repo_name": repo_id.split("/")[1],
                    "path": path,
                    "filename": Path(path).name,
                    "size": entry.get("size"),
                    "sha": entry.get("oid"),
                    "category": category,
                    "family": family,
                    "title": title,
                    "downloads": int(repo.get("downloads", 0) or 0),
                    "likes": int(repo.get("likes", 0) or 0),
                    "tags": repo.get("tags", []),
                    "last_modified": repo.get("lastModified"),
                }
                items.append(item)
                files_scanned += 1
                if files_scanned % 1500 == 0:
                    # Yield so index rebuild does not monopolize the event loop.
                    await asyncio.sleep(0)

    items = dedupe_items(items)

    def sort_key(item: Dict[str, object]) -> Tuple[int, str, int, int, str]:
        category = str(item.get("category", ""))
        category_idx = CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else 999
        family = str(item.get("family", "MISC"))
        downloads = int(item.get("downloads", 0) or 0)
        likes = int(item.get("likes", 0) or 0)
        title = str(item.get("title", "")).lower()
        return (category_idx, family, -downloads, -likes, title)

    items.sort(key=sort_key)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "generated_at": int(time.time()),
        "owners": list(owners),
        "limit_per_owner": int(limit_per_owner),
        "strict_filter": bool(strict_filter),
        "items": items,
        "errors": errors,
    }


def ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_cached_index() -> Optional[Dict[str, object]]:
    if not CACHE_PATH.exists():
        return None
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return None
        return payload
    except Exception:
        return None


def save_index(index: Dict[str, object]) -> None:
    ensure_cache_dir()
    with CACHE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)


def _is_cache_hit(
    cached: Optional[Dict[str, object]],
    owners: Sequence[str],
    limit_per_owner: int,
    strict_filter: bool,
) -> bool:
    if not cached:
        return False
    same_schema = int(cached.get("schema_version", 0) or 0) == CACHE_SCHEMA_VERSION
    same_owners = cached.get("owners") == list(owners)
    same_limit = int(cached.get("limit_per_owner", -1)) == int(limit_per_owner)
    same_strict = bool(cached.get("strict_filter", STRICT_FILTER_DEFAULT)) == bool(strict_filter)
    fresh = (time.time() - float(cached.get("generated_at", 0))) < CACHE_MAX_AGE_SECONDS
    return same_schema and same_owners and same_limit and same_strict and fresh


async def _acquire_index_build_lock() -> None:
    while True:
        acquired = INDEX_BUILD_LOCK.acquire(blocking=False)
        if acquired:
            return
        await asyncio.sleep(0.05)


async def load_or_build_index(
    owners: Sequence[str],
    limit_per_owner: int,
    refresh: bool,
    strict_filter: bool,
) -> Dict[str, object]:
    cached = load_cached_index()
    if not refresh and _is_cache_hit(cached, owners, limit_per_owner, strict_filter):
        return cached

    await _acquire_index_build_lock()
    try:
        cached = load_cached_index()
        if not refresh and _is_cache_hit(cached, owners, limit_per_owner, strict_filter):
            return cached
        index = await build_index(owners, limit_per_owner, strict_filter)
        save_index(index)
        return index
    finally:
        INDEX_BUILD_LOCK.release()


def get_tab_order(items: Sequence[Dict[str, object]]) -> List[str]:
    categories = {str(item.get("category", "")) for item in items}
    ordered = [category for category in CATEGORY_ORDER if category in categories]
    extras = sorted(categories - set(ordered))
    return ordered + extras


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\\\|?*]+', "_", name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or "model.safetensors"


def safe_filename_with_ext(name: str, extension: str) -> str:
    base = sanitize_filename(name)
    ext = Path(base).suffix
    if ext:
        return base
    if not extension:
        extension = ".safetensors"
    if not extension.startswith("."):
        extension = f".{extension}"
    return f"{base}{extension}"


def choose_extension(item: Dict[str, object]) -> str:
    suffix = Path(str(item.get("filename", ""))).suffix.lower()
    if suffix:
        return suffix
    return ".safetensors"


def models_root() -> Path:
    return Path(folder_paths.models_dir).resolve()


def preview_destination(item: Dict[str, object]) -> str:
    root = models_root()
    category = str(item.get("category", "checkpoints"))
    family = str(item.get("family", "MISC"))
    filename = sanitize_filename(str(item.get("filename", "")))
    if not Path(filename).suffix:
        filename = safe_filename_with_ext(filename, choose_extension(item))
    return str(root / category / family / filename)


def build_installed_lookup(items: Sequence[Dict[str, object]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    root = models_root()
    targets: set[Tuple[str, str]] = set()
    for item in items:
        category = str(item.get("category", "checkpoints"))
        family = str(item.get("family", "MISC"))
        targets.add((category, family))

    lookup: Dict[Tuple[str, str], Dict[str, str]] = {}
    for category, family in targets:
        directory = root / category / family
        file_map: Dict[str, str] = {}
        if directory.exists() and directory.is_dir():
            try:
                for entry in directory.iterdir():
                    if entry.is_file():
                        file_map[entry.name.lower()] = str(entry)
            except OSError:
                file_map = {}
        lookup[(category, family)] = file_map
    return lookup


def installed_path_for_item(item: Dict[str, object], lookup: Dict[Tuple[str, str], Dict[str, str]]) -> Optional[str]:
    category = str(item.get("category", "checkpoints"))
    family = str(item.get("family", "MISC"))
    filename = sanitize_filename(str(item.get("filename", "")))
    extension = choose_extension(item)
    if not Path(filename).suffix:
        filename = safe_filename_with_ext(filename, extension)
    lowered = filename.lower()
    directory = models_root() / category / family
    files = lookup.get((category, family), {})
    expected_size = item.get("size")
    expected_size_int = expected_size if isinstance(expected_size, int) and expected_size > 0 else None
    direct = files.get(lowered)
    if direct:
        if expected_size_int is not None:
            try:
                if Path(direct).stat().st_size != expected_size_int:
                    direct = None
            except OSError:
                direct = None
        if direct is None:
            pass
        else:
            return direct

    stem = Path(lowered).stem
    suffix = Path(lowered).suffix
    for existing_name, existing_path in files.items():
        if existing_name.startswith(f"{stem}__") and existing_name.endswith(suffix):
            if expected_size_int is not None:
                try:
                    if Path(existing_path).stat().st_size != expected_size_int:
                        continue
                except OSError:
                    continue
            return existing_path
    return None


def build_destination(
    item: Dict[str, object],
    reserved_targets: set[str],
    create_dirs: bool,
) -> Path:
    root = models_root()
    category = str(item.get("category", "checkpoints"))
    family = str(item.get("family", "MISC"))
    repo_name = str(item.get("repo_name", "repo"))
    target_dir = root / category / family
    if create_dirs:
        target_dir.mkdir(parents=True, exist_ok=True)

    extension = choose_extension(item)
    base_name = sanitize_filename(str(item.get("filename", "")))
    if not Path(base_name).suffix:
        base_name = safe_filename_with_ext(base_name, extension)

    candidate = target_dir / base_name
    norm = str(candidate).lower()
    if norm not in reserved_targets and not candidate.exists():
        reserved_targets.add(norm)
        return candidate

    if candidate.exists():
        item_size = item.get("size")
        if isinstance(item_size, int) and item_size > 0:
            try:
                if candidate.stat().st_size == item_size:
                    reserved_targets.add(norm)
                    return candidate
            except OSError:
                pass

    stem = Path(base_name).stem
    suffix = Path(base_name).suffix
    repo_hint = sanitize_filename(repo_name)
    dedupe_name = f"{stem}__{repo_hint}{suffix}"
    dedupe_path = target_dir / dedupe_name
    dedupe_norm = str(dedupe_path).lower()
    dedupe_idx = 2
    while dedupe_norm in reserved_targets or dedupe_path.exists():
        dedupe_name = f"{stem}__{repo_hint}_{dedupe_idx}{suffix}"
        dedupe_path = target_dir / dedupe_name
        dedupe_norm = str(dedupe_path).lower()
        dedupe_idx += 1
    reserved_targets.add(dedupe_norm)
    return dedupe_path


def download_url(item: Dict[str, object]) -> str:
    repo_id = str(item.get("repo_id", ""))
    path = str(item.get("path", ""))
    revision = str(item.get("repo_revision", "main") or "main").strip() or "main"
    quoted_rev = quote(revision, safe="")
    quoted_path = quote(path, safe="/")
    return f"{HF_WEB}/{repo_id}/resolve/{quoted_rev}/{quoted_path}?download=1"


async def validate_hf_token_async(token: str) -> Tuple[bool, str]:
    if not token:
        return False, "No token provided"
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "comfyui-hf-model-downloader/1.0",
    }
    timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_connect=10, sock_read=20)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{HF_WEB}/api/whoami-v2", headers=headers) as response:
                if response.status >= 400:
                    details = (await response.text())[:240]
                    return False, details or f"HTTP {response.status}"
                raw = await response.text()
                payload = json.loads(raw)
    except aiohttp.ClientError as exc:
        return False, str(exc)
    except json.JSONDecodeError as exc:
        return False, f"Invalid JSON from token check: {exc}"
    except Exception as exc:
        return False, str(exc)
    identity = payload.get("name") or payload.get("fullname") or "authenticated user"
    return True, f"authenticated as {identity}"


def validate_hf_token(token: str) -> Tuple[bool, str]:
    """Validate HF token in a dedicated event loop.

    Intended for background/worker threads. Do not call from code already
    running on the ComfyUI aiohttp event loop (use ``validate_hf_token_async``).
    """
    return asyncio.run(validate_hf_token_async(token))


def _set_job(job_id: str, **fields: object) -> None:
    with JOBS_LOCK:
        if job_id not in JOBS:
            return
        safe_fields = {key: copy.deepcopy(value) for key, value in fields.items()}
        JOBS[job_id].update(safe_fields)


def _job_snapshot(job_id: str) -> Optional[Dict[str, object]]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None
        return copy.deepcopy(job)


def _jobs_snapshot() -> List[Dict[str, object]]:
    with JOBS_LOCK:
        return [copy.deepcopy(job) for job in JOBS.values()]


def _set_job_process(job_id: str, process: subprocess.Popen) -> None:
    with JOBS_LOCK:
        JOB_PROCESSES[job_id] = process


def _get_job_process(job_id: str) -> Optional[subprocess.Popen]:
    with JOBS_LOCK:
        return JOB_PROCESSES.get(job_id)


def _pop_job_process(job_id: str) -> None:
    with JOBS_LOCK:
        JOB_PROCESSES.pop(job_id, None)


def _is_cancel_requested(job_id: str) -> bool:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return False
        return bool(job.get("cancel_requested"))


def _append_job_log(job_id: str, line: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        existing = job.get("logs", [])
        logs: List[str]
        if isinstance(existing, list):
            logs = [str(entry) for entry in existing]
        else:
            logs = []
        logs.append(str(line))
        job["logs"] = logs[-120:]


def _create_aria2_queue(items: Sequence[Dict[str, object]], token: Optional[str]) -> Tuple[str, List[str]]:
    reserved_targets: set[str] = set()
    queue_file = tempfile.NamedTemporaryFile(
        prefix="hf_model_downloader_",
        suffix=".aria2",
        mode="w",
        encoding="utf-8",
        delete=False,
    )
    targets: List[str] = []
    for item in items:
        destination = build_destination(item, reserved_targets=reserved_targets, create_dirs=True)
        targets.append(str(destination))
        queue_file.write(f"{download_url(item)}\n")
        queue_file.write(f"  dir={destination.parent}\n")
        queue_file.write(f"  out={destination.name}\n")
        queue_file.write("  continue=true\n")
        if token:
            queue_file.write(f"  header=Authorization: Bearer {token}\n")
        queue_file.write("\n")
    queue_file.flush()
    queue_file.close()
    return queue_file.name, targets


def _expected_size(item: Dict[str, object]) -> Optional[int]:
    size = item.get("size")
    if isinstance(size, int) and size > 0:
        return size
    return None


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pick_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _normalized_target(path: str) -> str:
    return str(Path(path)).lower()


def _build_file_progress_entries(items: Sequence[Dict[str, object]], targets: Sequence[str]) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for idx, (item, target) in enumerate(zip(items, targets)):
        total_bytes = _expected_size(item) or 0
        entries.append(
            {
                "index": idx,
                "item_id": str(item.get("id", "")),
                "title": str(item.get("title", item.get("filename", "model"))),
                "filename": str(item.get("filename", Path(target).name)),
                "target": str(target),
                "status": "queued",
                "progress": 0.0,
                "downloaded_bytes": 0,
                "total_bytes": total_bytes,
                "speed_bps": 0,
                "error": None,
            }
        )
    return entries


def _aria2_rpc_call(rpc_url: str, secret: str, method: str, params: List[object]) -> object:
    payload = {
        "jsonrpc": "2.0",
        "id": "hfmd",
        "method": f"aria2.{method}",
        "params": [f"token:{secret}", *params],
    }
    req = UrlRequest(
        rpc_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with sync_urlopen(req, timeout=2.5) as response:
        raw = response.read().decode("utf-8")
    decoded = json.loads(raw)
    if isinstance(decoded, dict) and "error" in decoded:
        raise RuntimeError(f"aria2 rpc error: {decoded['error']}")
    return decoded.get("result") if isinstance(decoded, dict) else None


def _poll_aria2_items(rpc_url: str, secret: str) -> List[Dict[str, object]]:
    keys = ["gid", "status", "totalLength", "completedLength", "downloadSpeed", "files", "errorMessage"]
    active = _aria2_rpc_call(rpc_url, secret, "tellActive", [keys]) or []
    waiting = _aria2_rpc_call(rpc_url, secret, "tellWaiting", [0, 1000, keys]) or []
    stopped = _aria2_rpc_call(rpc_url, secret, "tellStopped", [0, 1000, keys]) or []
    merged: List[Dict[str, object]] = []
    for batch in (active, waiting, stopped):
        if not isinstance(batch, list):
            continue
        for item in batch:
            if isinstance(item, dict):
                merged.append(item)
    return merged


def _apply_aria2_progress(
    file_entries: List[Dict[str, object]],
    aria2_items: Sequence[Dict[str, object]],
) -> Tuple[int, int, int]:
    by_target = {_normalized_target(str(entry.get("target", ""))): entry for entry in file_entries}
    for info in aria2_items:
        files = info.get("files")
        if not isinstance(files, list) or not files:
            continue
        first = files[0] if isinstance(files[0], dict) else {}
        path = str(first.get("path", "")).strip()
        if not path:
            continue
        entry = by_target.get(_normalized_target(path))
        if not entry:
            continue
        raw_status = str(info.get("status", entry.get("status", "queued")))
        status = {
            "active": "downloading",
            "waiting": "queued",
            "paused": "paused",
            "complete": "complete",
            "error": "error",
            "removed": "cancelled",
        }.get(raw_status, raw_status)
        total = _safe_int(info.get("totalLength"), _safe_int(entry.get("total_bytes"), 0))
        completed = _safe_int(info.get("completedLength"), _safe_int(entry.get("downloaded_bytes"), 0))
        speed = _safe_int(info.get("downloadSpeed"), 0)
        if total > 0:
            completed = max(0, min(completed, total))
            progress = min(100.0, (completed / total) * 100.0)
        else:
            progress = 100.0 if status == "complete" else float(entry.get("progress", 0.0) or 0.0)
        entry["status"] = status
        entry["total_bytes"] = total
        entry["downloaded_bytes"] = completed
        entry["speed_bps"] = speed
        entry["progress"] = progress
        if status in {"error", "removed"}:
            entry["error"] = str(info.get("errorMessage", ""))[:300] if info.get("errorMessage") else None

    downloaded_bytes = 0
    total_bytes = 0
    completed_files = 0
    for entry in file_entries:
        total = _safe_int(entry.get("total_bytes"), 0)
        completed = _safe_int(entry.get("downloaded_bytes"), 0)
        status = str(entry.get("status", "queued"))
        if status == "complete" and total > 0:
            completed = total
            entry["downloaded_bytes"] = total
            entry["progress"] = 100.0
        if total > 0:
            downloaded_bytes += max(0, min(completed, total))
            total_bytes += total
        if status == "complete":
            completed_files += 1
    return downloaded_bytes, total_bytes, completed_files


def _snapshot_progress(items: Sequence[Dict[str, object]], targets: Sequence[str]) -> Tuple[int, int]:
    downloaded_bytes = 0
    completed = 0
    for item, target in zip(items, targets):
        expected = _expected_size(item)
        target_path = Path(target)
        marker_path = Path(f"{target}.aria2")
        marker_exists = marker_path.exists()
        target_exists = target_path.exists()
        current_size = 0
        try:
            if target_exists:
                current_size = target_path.stat().st_size
        except OSError:
            current_size = 0

        if expected is not None:
            observed = min(current_size, expected)
            if marker_exists and observed >= expected:
                # aria2 file pre-allocation can report full file size instantly.
                observed = max(0, expected - 1)
            downloaded_bytes += observed
            if target_exists and not marker_exists and current_size >= expected:
                completed += 1
        else:
            downloaded_bytes += current_size
            if target_exists and not marker_exists:
                completed += 1
    return downloaded_bytes, completed


def _verify_download_file(item: Dict[str, object], target: str) -> Tuple[bool, Optional[str]]:
    target_path = Path(target)
    marker_path = Path(f"{target}.aria2")
    if marker_path.exists():
        return False, "aria2 control file still present (incomplete download)"
    if not target_path.is_file():
        return False, "file missing"
    expected = _expected_size(item)
    try:
        actual = target_path.stat().st_size
    except OSError as exc:
        return False, f"stat failed: {exc}"
    if expected is not None and actual != expected:
        return False, f"size mismatch (expected {expected} bytes, got {actual})"
    return True, None


def _run_download_worker(
    job_id: str,
    items: List[Dict[str, object]],
    max_concurrent_downloads: int,
    connections_per_download: int,
) -> None:
    _set_job(job_id, status="running", started_at=int(time.time()))
    token = get_hf_token()
    selected_owners = {str(item.get("owner", "")).lower() for item in items}
    try:
        if "black-forest-labs" in selected_owners and not token:
            raise RuntimeError(
                "black-forest-labs models are gated. Set HF_TOKEN/HUGGINGFACE_TOKEN or add .hf_token."
            )

        if token:
            valid, detail = validate_hf_token(token)
            if not valid:
                raise RuntimeError(f"HF token invalid/expired: {detail}")
            _append_job_log(job_id, f"Token check: {detail}")

        aria2 = shutil.which("aria2c")
        if not aria2:
            raise RuntimeError("aria2c not found in PATH. Install aria2 to download models.")

        queue_file, targets = _create_aria2_queue(items, token)
        file_entries = _build_file_progress_entries(items, targets)
        total_bytes = sum(size for size in (_expected_size(item) for item in items) if size is not None)
        _set_job(
            job_id,
            total=len(items),
            completed=0,
            targets=targets,
            files=file_entries,
            total_bytes=total_bytes,
            downloaded_bytes=0,
            progress=0.0,
        )
        rpc_port = _pick_free_local_port()
        rpc_secret = secrets.token_hex(16)
        rpc_url = f"http://127.0.0.1:{rpc_port}/jsonrpc"

        cmd = [
            aria2,
            f"--input-file={queue_file}",
            "--allow-overwrite=false",
            "--auto-file-renaming=false",
            "--continue=true",
            "--check-integrity=true",
            "--file-allocation=none",
            "--summary-interval=0",
            "--download-result=hide",
            "--console-log-level=warn",
            "--enable-rpc=true",
            "--rpc-listen-all=false",
            f"--rpc-listen-port={rpc_port}",
            f"--rpc-secret={rpc_secret}",
            f"--max-concurrent-downloads={max(1, int(max_concurrent_downloads))}",
            f"--max-connection-per-server={max(1, int(connections_per_download))}",
            f"--split={max(1, int(connections_per_download))}",
        ]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _set_job_process(job_id, process)
        rpc_failed_logged = False
        while True:
            downloaded_bytes = 0
            completed = 0
            rpc_used = False
            try:
                aria2_items = _poll_aria2_items(rpc_url, rpc_secret)
                if aria2_items:
                    rpc_downloaded, rpc_total, rpc_completed = _apply_aria2_progress(file_entries, aria2_items)
                    downloaded_bytes = rpc_downloaded
                    if rpc_total > 0:
                        total_bytes = rpc_total
                    completed = rpc_completed
                    rpc_used = True
            except Exception as rpc_exc:
                if not rpc_failed_logged:
                    _append_job_log(job_id, f"aria2 rpc fallback enabled: {rpc_exc}")
                    rpc_failed_logged = True

            if not rpc_used:
                downloaded_bytes, completed = _snapshot_progress(items, targets)
                for item, target, entry in zip(items, targets, file_entries):
                    expected = _expected_size(item) or _safe_int(entry.get("total_bytes"), 0)
                    target_path = Path(target)
                    marker_exists = Path(f"{target}.aria2").exists()
                    current_size = 0
                    try:
                        if target_path.exists():
                            current_size = target_path.stat().st_size
                    except OSError:
                        current_size = 0
                    entry["total_bytes"] = expected
                    entry["downloaded_bytes"] = min(current_size, expected) if expected > 0 else current_size
                    if target_path.exists() and not marker_exists:
                        entry["status"] = "complete"
                        entry["progress"] = 100.0
                    elif target_path.exists():
                        entry["status"] = "downloading"
                        if expected > 0:
                            entry["progress"] = min(100.0, (min(current_size, expected) / expected) * 100.0)
                    else:
                        entry["status"] = "queued"

            current_job = _job_snapshot(job_id) or {}
            prev_downloaded = int(current_job.get("downloaded_bytes", 0) or 0)
            prev_completed = int(current_job.get("completed", 0) or 0)
            downloaded_bytes = max(downloaded_bytes, prev_downloaded)
            completed = max(completed, prev_completed)
            if total_bytes > 0:
                progress = min(100.0, (downloaded_bytes / total_bytes) * 100.0)
            else:
                progress = min(100.0, (completed / max(1, len(items))) * 100.0)
            _set_job(
                job_id,
                completed=completed,
                downloaded_bytes=downloaded_bytes,
                total_bytes=total_bytes,
                progress=progress,
                files=file_entries,
            )
            if process.poll() is not None:
                break
            time.sleep(0.6)

        if process.returncode != 0:
            if _is_cancel_requested(job_id):
                raise RuntimeError("cancelled")
            raise RuntimeError(f"aria2 exited with code {process.returncode}")

        ok_count = 0
        failures: List[str] = []
        for item, target, entry in zip(items, targets, file_entries):
            ok, verify_error = _verify_download_file(item, target)
            if ok:
                ok_count += 1
                entry["status"] = "complete"
                entry["progress"] = 100.0
                expected = _expected_size(item)
                row_total = expected if expected is not None else _safe_int(entry.get("total_bytes"), 0)
                if row_total > 0:
                    entry["total_bytes"] = row_total
                    entry["downloaded_bytes"] = row_total
            else:
                entry["status"] = "error"
                entry["error"] = (verify_error or "verification failed")[:300]
                failures.append(str(entry.get("filename", Path(target).name)))

        downloaded_bytes, completed = _snapshot_progress(items, targets)
        completed = max(completed, ok_count)

        if failures:
            if total_bytes > 0:
                progress = min(100.0, (downloaded_bytes / total_bytes) * 100.0)
            else:
                progress = min(100.0, (ok_count / max(1, len(items))) * 100.0)
            _set_job(
                job_id,
                status="error",
                completed=ok_count,
                downloaded_bytes=downloaded_bytes,
                total_bytes=total_bytes,
                progress=progress,
                finished_at=int(time.time()),
                error=f"{len(failures)} file(s) failed verification.",
                message=(
                    f"Verified {ok_count}/{len(items)} file(s). Failures: {', '.join(failures[:5])}"
                    + ("…" if len(failures) > 5 else "")
                ),
                cancel_requested=False,
                files=file_entries,
            )
            _append_job_log(job_id, f"Verification failures: {len(failures)} file(s).")
        else:
            _set_job(
                job_id,
                status="done",
                completed=max(completed, len(items)),
                downloaded_bytes=max(downloaded_bytes, total_bytes),
                total_bytes=total_bytes,
                progress=100.0,
                finished_at=int(time.time()),
                message=f"Downloaded {len(items)} model(s).",
                cancel_requested=False,
                files=file_entries,
            )
    except Exception as exc:
        downloaded_bytes = 0
        total_bytes = 0
        current = _job_snapshot(job_id) or {}
        downloaded_bytes = int(current.get("downloaded_bytes", 0) or 0)
        total_bytes = int(current.get("total_bytes", 0) or 0)
        progress = 0.0
        if total_bytes > 0:
            progress = min(100.0, (downloaded_bytes / total_bytes) * 100.0)
        if str(exc) == "cancelled" or _is_cancel_requested(job_id):
            if "file_entries" in locals() and isinstance(file_entries, list):
                for entry in file_entries:
                    status = str(entry.get("status", "queued"))
                    if status not in {"complete", "error"}:
                        entry["status"] = "cancelled"
            _set_job(
                job_id,
                status="cancelled",
                finished_at=int(time.time()),
                error=None,
                message="Download cancelled.",
                progress=progress,
                cancel_requested=False,
                files=file_entries if "file_entries" in locals() else None,
            )
            return
        if "file_entries" in locals() and isinstance(file_entries, list):
            for entry in file_entries:
                status = str(entry.get("status", "queued"))
                if status not in {"complete", "cancelled"}:
                    entry["status"] = "error"
                    if not entry.get("error"):
                        entry["error"] = str(exc)[:300]
        _set_job(
            job_id,
            status="error",
            finished_at=int(time.time()),
            error=str(exc),
            message="Download failed.",
            progress=progress,
            files=file_entries if "file_entries" in locals() else None,
        )
        _append_job_log(job_id, f"Error: {exc}")
    finally:
        _pop_job_process(job_id)
        _prune_finished_jobs()
        try:
            if "queue_file" in locals() and queue_file:
                Path(queue_file).unlink(missing_ok=True)
        except Exception:
            pass


def _get_index_owners_limit_and_filter(request: web.Request) -> Tuple[List[str], int, bool]:
    raw_owners = request.query.get("owners", "")
    if raw_owners.strip():
        owners = normalize_owner_list(raw_owners.split(","))
    else:
        owners = list(DEFAULT_OWNERS)
    if not owners:
        owners = list(DEFAULT_OWNERS)
    limit = parse_int(
        request.query.get("limit_per_owner"),
        default=DEFAULT_LIMIT_PER_OWNER,
        min_value=1,
        max_value=120,
    )
    strict_filter = parse_bool(request.query.get("strict_filter"))
    if "strict_filter" not in request.query:
        strict_filter = parse_bool(request.query.get("strict")) if "strict" in request.query else STRICT_FILTER_DEFAULT
    return owners, limit, strict_filter


def _serialize_index(index: Dict[str, object]) -> Dict[str, object]:
    items = index.get("items", [])
    if not isinstance(items, list):
        items = []

    installed_lookup = build_installed_lookup([item for item in items if isinstance(item, dict)])
    category_counts: Dict[str, int] = {}
    installed_count = 0
    payload_items: List[Dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", "checkpoints"))
        category_counts[category] = category_counts.get(category, 0) + 1
        payload = dict(item)
        payload["target_preview"] = preview_destination(item)
        installed_path = installed_path_for_item(item, installed_lookup)
        payload["installed"] = bool(installed_path)
        payload["installed_path"] = installed_path
        if installed_path:
            installed_count += 1
        payload_items.append(payload)

    return {
        "ok": True,
        "schema_version": index.get("schema_version", CACHE_SCHEMA_VERSION),
        "generated_at": index.get("generated_at"),
        "owners": index.get("owners", []),
        "default_owners": list(DEFAULT_OWNERS),
        "limit_per_owner": index.get("limit_per_owner"),
        "strict_filter": bool(index.get("strict_filter", STRICT_FILTER_DEFAULT)),
        "item_count": len(payload_items),
        "installed_count": installed_count,
        "categories": get_tab_order(payload_items),
        "category_counts": category_counts,
        "errors": index.get("errors", []),
        "items": payload_items,
    }


routes = PromptServer.instance.routes


@routes.get("/hf-model-downloader/settings")
async def hf_model_downloader_settings(request: web.Request) -> web.Response:
    token, source = _token_with_source()
    configured = bool(token)
    return web.json_response(
        {
            "ok": True,
            "token_configured": configured,
            "token_source": source,
            "token_hint": mask_token(token) if configured else "",
            "token_file": str(TOKEN_CANDIDATE_PATHS[0]),
        }
    )


@routes.post("/hf-model-downloader/token")
async def hf_model_downloader_token(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)

    token = str(payload.get("token", "")).strip()
    if not token:
        removed = False
        for path in TOKEN_CANDIDATE_PATHS:
            try:
                if path.exists():
                    path.unlink()
                    removed = True
            except OSError:
                continue
        return web.json_response(
            {
                "ok": True,
                "token_configured": False,
                "message": "HF token cleared." if removed else "No saved token to clear.",
            }
        )

    valid, detail = await validate_hf_token_async(token)
    if not valid:
        return web.json_response({"ok": False, "error": f"Token validation failed: {detail}"}, status=400)

    target_path = TOKEN_CANDIDATE_PATHS[0]
    try:
        target_path.write_text(token + "\n", encoding="utf-8")
        os.chmod(target_path, 0o600)
    except OSError as exc:
        return web.json_response({"ok": False, "error": f"Failed to save token: {exc}"}, status=500)

    return web.json_response(
        {
            "ok": True,
            "token_configured": True,
            "token_source": f"file:{target_path}",
            "token_hint": mask_token(token),
            "message": f"HF token saved ({detail}).",
        }
    )


@routes.get("/hf-model-downloader/index")
async def hf_model_downloader_index(request: web.Request) -> web.Response:
    owners, limit, strict_filter = _get_index_owners_limit_and_filter(request)
    refresh = parse_bool(request.query.get("refresh"))
    try:
        index = await load_or_build_index(owners, limit, refresh, strict_filter)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=500)
    return web.json_response(_serialize_index(index))


@routes.post("/hf-model-downloader/download")
async def hf_model_downloader_download(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)

    # Live-browse selections arrive as fully formed items, because they are not in the
    # curated index at all. Indexed selections still arrive as bare ids.
    live_items = payload.get("items")
    if isinstance(live_items, list) and live_items:
        if len(live_items) > 500:
            return web.json_response({"ok": False, "error": "Too many items selected (max 500)."}, status=400)
        selected = normalize_live_items(live_items)
        if not selected:
            return web.json_response(
                {"ok": False, "error": "No downloadable files in the submitted items."}, status=400
            )
        return _start_download_job(selected, payload)

    ids = payload.get("ids", [])
    if not isinstance(ids, list) or not ids:
        return web.json_response({"ok": False, "error": "Request body must include a non-empty ids list."}, status=400)
    if len(ids) > 500:
        return web.json_response({"ok": False, "error": "Too many items selected (max 500)."}, status=400)

    owners = payload.get("owners")
    if isinstance(owners, list) and owners:
        normalized_owners = normalize_owner_list([str(owner) for owner in owners])
    else:
        normalized_owners = list(DEFAULT_OWNERS)
    if not normalized_owners:
        normalized_owners = list(DEFAULT_OWNERS)
    limit = parse_int(
        str(payload.get("limit_per_owner", DEFAULT_LIMIT_PER_OWNER)),
        default=DEFAULT_LIMIT_PER_OWNER,
        min_value=1,
        max_value=120,
    )
    if "strict_filter" in payload:
        strict_filter = parse_bool(str(payload.get("strict_filter")))
    else:
        strict_filter = STRICT_FILTER_DEFAULT

    try:
        index = await load_or_build_index(normalized_owners, limit, False, strict_filter)
    except Exception as exc:
        return web.json_response({"ok": False, "error": f"Failed to load index: {exc}"}, status=500)

    items = index.get("items", [])
    if not isinstance(items, list):
        items = []
    by_id = {str(item.get("id")): item for item in items if isinstance(item, dict)}
    selected = [by_id[item_id] for item_id in ids if item_id in by_id]
    if not selected:
        return web.json_response({"ok": False, "error": "Selected IDs were not found in the current index."}, status=400)

    return _start_download_job(selected, payload)


def normalize_live_items(rows: Sequence[object]) -> List[Dict[str, object]]:
    """Accept browser-supplied file rows, keeping only what the worker needs."""
    cleaned: List[Dict[str, object]] = []
    seen: set[Tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        repo_id = str(row.get("repo_id") or "").strip().strip("/")
        path = str(row.get("path") or "").strip().lstrip("/")
        if not repo_id or repo_id.count("/") != 1 or not path:
            continue
        if ".." in Path(path).parts:
            continue
        if Path(path).suffix.lower() not in MODEL_EXTENSIONS:
            continue
        key = (repo_id.lower(), path.lower())
        if key in seen:
            continue
        seen.add(key)

        size = row.get("size")
        try:
            size = int(size or 0)
        except (TypeError, ValueError):
            size = 0
        category = str(row.get("category") or "checkpoints")
        if category not in VALID_CATEGORIES:
            category = "checkpoints"
        revision = str(row.get("repo_revision") or "main").strip() or "main"
        cleaned.append(
            {
                "id": f"live::{repo_id}::{path}",
                "repo_id": repo_id,
                "repo_name": repo_id.split("/")[-1],
                "repo_revision": revision,
                "path": path,
                "filename": Path(path).name,
                "size": size,
                "category": category,
                "family": str(row.get("family") or "MISC"),
                "title": str(row.get("title") or Path(path).name),
            }
        )
    return cleaned


def _start_download_job(selected: Sequence[Dict[str, object]], payload: Dict[str, object]) -> web.Response:
    max_cd = parse_int(
        str(payload.get("max_concurrent_downloads", 8)),
        default=8,
        min_value=1,
        max_value=32,
    )
    connections = parse_int(
        str(payload.get("connections_per_download", 16)),
        default=16,
        min_value=1,
        max_value=32,
    )

    selected = list(selected)
    _prune_finished_jobs()
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "total": len(selected),
            "completed": 0,
            "total_bytes": 0,
            "downloaded_bytes": 0,
            "progress": 0.0,
            "created_at": int(time.time()),
            "message": "Queued for download.",
            "error": None,
            "cancel_requested": False,
            "logs": [],
            "targets": [],
            "files": [],
        }

    worker = threading.Thread(
        target=_run_download_worker,
        args=(job_id, selected, max_cd, connections),
        daemon=True,
    )
    worker.start()
    return web.json_response({"ok": True, "job_id": job_id, "total": len(selected)})


@routes.get("/hf-model-downloader/status")
async def hf_model_downloader_status(request: web.Request) -> web.Response:
    job_id = request.query.get("job_id", "").strip()
    if not job_id:
        return web.json_response({"ok": False, "error": "Missing job_id query parameter."}, status=400)
    payload = _job_snapshot(job_id)
    if not payload:
        return web.json_response({"ok": False, "error": f"Job not found: {job_id}"}, status=404)
    payload["ok"] = True
    return web.json_response(payload)


@routes.get("/hf-model-downloader/jobs")
async def hf_model_downloader_jobs(request: web.Request) -> web.Response:
    limit = parse_int(request.query.get("limit"), default=25, min_value=1, max_value=100)
    jobs = _jobs_snapshot()
    jobs.sort(key=lambda job: int(job.get("created_at", 0) or 0), reverse=True)
    trimmed: List[Dict[str, object]] = []
    for job in jobs[:limit]:
        logs = job.get("logs")
        if isinstance(logs, list):
            job["logs"] = logs[-20:]
        trimmed.append(job)
    return web.json_response({"ok": True, "jobs": trimmed})


@routes.post("/hf-model-downloader/cancel")
async def hf_model_downloader_cancel(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)

    job_id = str(payload.get("job_id", "")).strip()
    if not job_id:
        return web.json_response({"ok": False, "error": "Missing job_id in request body."}, status=400)

    job = _job_snapshot(job_id)
    if not job:
        return web.json_response({"ok": False, "error": f"Job not found: {job_id}"}, status=404)

    status = str(job.get("status", "")).lower()
    if status in {"done", "error", "cancelled"}:
        return web.json_response({"ok": True, "job_id": job_id, "status": status, "message": "Job already finished."})

    _set_job(
        job_id,
        cancel_requested=True,
        message="Cancelling download...",
    )
    process = _get_job_process(job_id)
    if process and process.poll() is None:
        try:
            process.terminate()
        except Exception:
            pass

    return web.json_response({"ok": True, "job_id": job_id, "status": "cancelling"})
