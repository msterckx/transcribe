#!/usr/bin/env python3
"""
Runpod Serverless noisy-audio transcription worker.

Built for very noisy short-form recordings (VHF radio / ATC-style traffic).
Generic Whisper hallucinates badly on this kind of audio (wrong language,
invented words). The default model here is a Whisper large-v3 checkpoint
fine-tuned on the ATCO2 corpus of real air-traffic-control radio audio,
which is dramatically more accurate on this domain. See README.md for the
comparison against generic Whisper that justified this choice.

Storage backend (auto-detected, in this priority order):

1. Network Volume mounted at /runpod-volume. If the endpoint has one
   attached, RunPod mounts it automatically in every worker -- this is
   plain local disk, no configuration needed, and is always preferred when
   present since it's faster than going over the S3 API.
2. S3-compatible bucket, used only when there is no local mount. Useful
   when you deliberately don't attach a Network Volume, since attaching one
   pins the endpoint to that volume's data center region and can limit
   which GPUs are available. Set:
      RUNPOD_S3_BUCKET
      RUNPOD_S3_ENDPOINT
      RUNPOD_S3_ACCESS_KEY_ID
      RUNPOD_S3_SECRET_ACCESS_KEY
      RUNPOD_S3_REGION            (optional, defaults to "us-east-1")

Either way, "source"/"sources" are keys into whichever backend is active,
and outputs are written back to the same one. This is the same shared
volume/bucket already used by the image-upscale and voicestudio workers on
this account, so "transcribe/..." key prefixes are used here to avoid
colliding with their "upscale/..." / "voice-studio/..." keys.

Input object example:
{
  "input": {
    "source": "transcribe/input/clip.wav",
    "language": "en",
    "model": "atc-large-v3"
  }
}

"source" also accepts a list, and "sources" is available as an alias, so a
single job can transcribe a batch:
{
  "input": {
    "sources": ["transcribe/input/a.wav", "transcribe/input/b.wav"]
  }
}

Audio can also be sent inline in the request instead of uploaded first,
via "audio_base64" (or "audios_base64" for a batch) -- a plain base64
string, a "data:audio/wav;base64,..." URI, or {"data": ..., "name": ...}
for a custom output label:
{
  "input": {
    "audio_base64": "UklGRi..."
  }
}
"source"/"sources" and "audio_base64"/"audios_base64" can be mixed in one
batch request.

Unless "save_output" is false, a plain-text transcript is also written to
"transcribe/output/<job-id>/<filename>.txt" on whichever backend is active
(this still happens for inline-audio items -- only the input audio skips
storage, not the output transcript).

Results are always returned in the "results" list, in the same order as the
requested sources; sources that fail are reported in "failed" while the rest
of the batch still completes.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import runpod
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Persistent storage
# ---------------------------------------------------------------------------

LOCAL_TEST = os.environ.get("LOCAL_TEST") == "1"

VOLUME_ROOT = Path(os.environ.get("RUNPOD_VOLUME_ROOT", "/runpod-volume"))
VOLUME_MOUNTED = VOLUME_ROOT.is_dir()

S3_BUCKET = os.environ.get("RUNPOD_S3_BUCKET")
S3_ENDPOINT = os.environ.get("RUNPOD_S3_ENDPOINT")
S3_REGION = os.environ.get("RUNPOD_S3_REGION", "us-east-1")
S3_ACCESS_KEY_ID = os.environ.get("RUNPOD_S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("RUNPOD_S3_SECRET_ACCESS_KEY")

# A mounted Network Volume is local disk -- faster and simpler than the S3
# API, and available in every worker automatically once attached to the
# endpoint. Prefer it whenever it's actually present, even if RUNPOD_S3_*
# vars are also set (e.g. left over from before a volume was attached).
# S3 is only the active backend when there is no local mount to fall back to.
USE_S3 = (
    bool(S3_BUCKET and S3_ENDPOINT and S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY)
    and not VOLUME_MOUNTED
)

_s3_client = None


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        import boto3
        from botocore.config import Config

        _s3_client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            region_name=S3_REGION,
            aws_access_key_id=S3_ACCESS_KEY_ID,
            aws_secret_access_key=S3_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
        )
    return _s3_client


# Only set up a persistent Hugging Face cache when there is somewhere
# persistent to put it. /runpod-volume may not exist at all when the
# worker is configured for S3-only access (no filesystem volume mount) --
# unconditionally mkdir-ing under it would crash the worker on cold start.
if not LOCAL_TEST and not USE_S3 and VOLUME_MOUNTED:
    DATA_ROOT = VOLUME_ROOT / "transcribe"
    HF_HOME = DATA_ROOT / "cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))
    (HF_HOME / "hub").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Baked into the image at build time (see Dockerfile.txt) so the default
# model needs no download on cold start.
BUILTIN_MODEL_DIR = Path(os.environ.get("BUILTIN_MODEL_DIR", "/app/models/atc-large-v3"))
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "atc-large-v3")

# Short aliases for known-good checkpoints. Anything else passed in "model"
# is treated as an arbitrary faster-whisper/CTranslate2-compatible HF repo
# id and downloaded on first use into the persistent HF cache above.
MODEL_ALIASES = {
    "atc-large-v3": str(BUILTIN_MODEL_DIR),
    "large-v3": "large-v3",
    "medium": "medium",
    "small": "small",
}

SUPPORTED_INPUT_EXTENSIONS = {
    ".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".wma", ".opus",
}

_MODELS: dict[str, WhisperModel] = {}


def _ensure_preprocessor_config(model_dir: Path, repo_hint: str) -> None:
    """
    Some community CTranslate2 conversions of Whisper large-v3 forget to
    ship preprocessor_config.json. faster-whisper then defaults to 80 mel
    bins, but v3 checkpoints expect 128, which fails with a shape mismatch
    at inference time. Patch it in if it's missing, guessing bin count from
    the repo/model name (large-v3 uses 128, everything else uses 80).
    """
    config_path = model_dir / "preprocessor_config.json"
    if config_path.exists():
        return

    feature_size = 128 if "v3" in repo_hint.lower() else 80

    config_path.write_text(json.dumps({
        "chunk_length": 30,
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": feature_size,
        "hop_length": 160,
        "n_fft": 400,
        "n_samples": 480000,
        "nb_max_frames": 3000,
        "padding_side": "right",
        "padding_value": 0.0,
        "processor_class": "WhisperProcessor",
        "return_attention_mask": False,
        "sampling_rate": 16000,
    }))


def get_model(model_id: str) -> WhisperModel:
    if model_id in _MODELS:
        return _MODELS[model_id]

    resolved = MODEL_ALIASES.get(model_id, model_id)

    if Path(resolved).is_dir():
        _ensure_preprocessor_config(Path(resolved), resolved)

    device = "cuda" if os.environ.get("FORCE_CPU") != "1" else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"Loading Whisper model '{model_id}' ({resolved}) on {device}...", flush=True)

    try:
        model = WhisperModel(resolved, device=device, compute_type=compute_type)
    except Exception:
        if device == "cuda":
            print("CUDA load failed, falling back to CPU", flush=True)
            model = WhisperModel(resolved, device="cpu", compute_type="int8")
        else:
            raise

    _MODELS[model_id] = model
    return model


# Preload the default model at worker startup so the first job doesn't pay
# the load cost.
if os.environ.get("SKIP_PRELOAD") != "1":
    get_model(DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# Audio loading / preprocessing
# ---------------------------------------------------------------------------

def decode_audio(data: bytes, label: str) -> tuple[np.ndarray, float]:
    """
    Decode any ffmpeg-readable audio bytes to mono float32 PCM at 16kHz,
    matching Whisper's expected input regardless of source format/rate.
    Fed via stdin so it works identically whether the bytes came from local
    disk or an S3 GetObject.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", "pipe:0",
        "-f", "f32le", "-ac", "1", "-ar", "16000",
        "-loglevel", "error", "-",
    ]
    proc = subprocess.run(cmd, input=data, capture_output=True, check=False)

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {label}: {proc.stderr.decode(errors='replace')}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    duration = len(audio) / 16000.0
    return audio, duration


def denoise(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Optional preprocessing for radio-style noise: a speech-band bandpass
    filter plus stationary-noise spectral gating.

    Disabled by default: on the ATC-tuned model this measurably *hurt*
    accuracy in testing (it clipped low-energy word endings and sometimes
    truncated short transmissions). Kept available for other noise profiles
    where it may help (e.g. hum, hiss on a model not trained on raw radio
    audio).
    """
    from scipy.signal import butter, sosfiltfilt
    import noisereduce as nr

    sos = butter(4, [300, 3400], btype="band", fs=sr, output="sos")
    filtered = sosfiltfilt(sos, audio)
    cleaned = nr.reduce_noise(y=filtered, sr=sr, stationary=True, prop_decrease=0.85)

    peak = float(np.max(np.abs(cleaned))) or 1.0
    return (cleaned / peak * 0.95).astype("float32")


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def validate_key(source_key: str) -> str:
    """
    Normalise and validate a source key. Used for both S3 object keys and
    Network Volume relative paths, since the safety requirements are the
    same either way: no escaping the bucket/volume, and a supported
    audio extension.
    """
    key = source_key.strip().lstrip("/")

    if not key or ".." in Path(key).parts:
        raise ValueError(f"invalid source key: {source_key!r}")

    suffix = Path(key).suffix.lower()
    if suffix not in SUPPORTED_INPUT_EXTENSIONS:
        raise ValueError(
            f"Unsupported source extension: {suffix}. "
            f"Supported: {sorted(SUPPORTED_INPUT_EXTENSIONS)}"
        )

    return key


def fetch_audio_bytes(source_key: str) -> bytes:
    """
    Read raw audio bytes for a source key from whichever backend is active:
    the S3-compatible bucket if RUNPOD_S3_* is configured, otherwise the
    Network Volume mount (or the current directory in LOCAL_TEST mode).
    """
    key = validate_key(source_key)

    if USE_S3:
        client = get_s3_client()
        try:
            obj = client.get_object(Bucket=S3_BUCKET, Key=key)
        except Exception as exc:
            error_code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                raise FileNotFoundError(f"Source audio not found in bucket {S3_BUCKET}: {key}") from exc
            raise
        return obj["Body"].read()

    root = Path.cwd() if LOCAL_TEST else VOLUME_ROOT
    source_path = (root / key).resolve()

    try:
        source_path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("source must stay inside the Network Volume") from exc

    if not source_path.is_file():
        raise FileNotFoundError(f"Source audio not found: {source_path}")

    return source_path.read_bytes()


def write_output(job_id: str, stem: str, text: str) -> str:
    """
    Write a plain-text transcript to whichever backend is active and return
    its key.
    """
    output_key = f"transcribe/output/{job_id}/{stem}.txt"
    body = (text + "\n").encode("utf-8")

    if USE_S3:
        get_s3_client().put_object(Bucket=S3_BUCKET, Key=output_key, Body=body)
        return output_key

    root = Path.cwd() if LOCAL_TEST else VOLUME_ROOT
    output_path = root / output_key
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(body)
    return output_key


MAX_INLINE_AUDIO_BYTES = 50 * 1024 * 1024  # 50MB decoded; these are short noisy clips, not long recordings


def decode_base64_audio(value: str) -> bytes:
    """Accept a plain base64 string or a data: URI and return raw bytes."""
    if value.startswith("data:"):
        _, _, value = value.partition(",")

    try:
        data = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError(f"invalid base64 audio data: {exc}") from exc

    if len(data) > MAX_INLINE_AUDIO_BYTES:
        raise ValueError(
            f"inline audio too large ({len(data)} bytes, max {MAX_INLINE_AUDIO_BYTES}); "
            'upload it to the storage backend and use "source" instead'
        )

    return data


def collect_items(job_input: dict) -> list[dict]:
    """
    Build an ordered list of work items from two independent input styles,
    which can be mixed in one batch:

    * "source"/"sources": object key(s) in the configured storage backend
      (Network Volume or S3 bucket).
    * "audio_base64"/"audios_base64": inline base64-encoded audio (a plain
      base64 string, a data: URI, or {"data": ..., "name": ...} for a
      custom label), needing no upload step at all.

    Each item is {"source": <label used in the response/output filename>,
    "kind": "key"|"inline", ...}.
    """
    items: list[dict] = []

    raw_sources = []
    for field in ("source", "sources"):
        value = job_input.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            raw_sources.append(value)
        elif isinstance(value, (list, tuple)):
            raw_sources.extend(value)
        else:
            raise ValueError(f'"{field}" must be a string or a list of strings')

    for value in raw_sources:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("every source must be a non-empty string")
        key = value.strip()
        items.append({"source": key, "kind": "key", "key": key})

    raw_audios = []
    for field in ("audio_base64", "audios_base64"):
        value = job_input.get(field)
        if value is None:
            continue
        if isinstance(value, (str, dict)):
            raw_audios.append(value)
        elif isinstance(value, (list, tuple)):
            raw_audios.extend(value)
        else:
            raise ValueError(f'"{field}" must be a string, an object, or a list of those')

    for index, value in enumerate(raw_audios, start=1):
        if isinstance(value, str):
            data_b64, name = value, None
        elif isinstance(value, dict):
            data_b64 = value.get("data")
            name = value.get("name")
            if not isinstance(data_b64, str) or not data_b64.strip():
                raise ValueError(f'audio item {index}: missing required field "data" (base64 string)')
        else:
            raise ValueError(f'audio item {index} must be a base64 string or an object with a "data" field')

        label = name.strip() if isinstance(name, str) and name.strip() else f"inline-{index}"
        items.append({"source": label, "kind": "inline", "data_b64": data_b64})

    if not items:
        raise ValueError(
            'Missing audio input: provide "source"/"sources" (object keys in the '
            'storage backend) or "audio_base64"/"audios_base64" (inline base64-'
            "encoded audio)"
        )

    # De-dup labels defensively so batch outputs don't collide (e.g. the same
    # key requested twice, or two inline items both named "clip.wav").
    seen: dict[str, int] = {}
    for item in items:
        base_label = item["source"]
        count = seen.get(base_label, 0)
        seen[base_label] = count + 1
        if count:
            item["source"] = f"{base_label}-{count}"

    return items


def fetch_item_bytes(item: dict) -> bytes:
    if item["kind"] == "inline":
        return decode_base64_audio(item["data_b64"])
    return fetch_audio_bytes(item["key"])


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_bytes(
    data: bytes,
    label: str,
    model: WhisperModel,
    language: str | None,
    vad_filter: bool,
    apply_denoise: bool,
    beam_size: int,
    initial_prompt: str | None,
    include_segments: bool,
) -> dict:
    audio, duration = decode_audio(data, label=label)

    if apply_denoise:
        audio = denoise(audio)

    segments_iter, info = model.transcribe(
        audio,
        language=language,
        vad_filter=vad_filter,
        beam_size=beam_size,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt,
    )

    segments = list(segments_iter)
    text = " ".join(s.text.strip() for s in segments).strip()

    result = {
        "text": text,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration_seconds": round(duration, 2),
    }

    if include_segments:
        result["segments"] = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments
        ]

    return result


def handler(job: dict) -> dict:
    job_input = job.get("input") or {}

    items = collect_items(job_input)

    model_id = str(job_input.get("model", DEFAULT_MODEL))
    language = job_input.get("language", "en")
    if language is not None:
        language = str(language)

    vad_filter = bool(job_input.get("vad_filter", False))
    apply_denoise = bool(job_input.get("denoise", False))
    beam_size = int(job_input.get("beam_size", 5))
    initial_prompt = job_input.get("initial_prompt")
    include_segments = bool(job_input.get("segments", False))
    save_output = bool(job_input.get("save_output", True))

    model = get_model(model_id)

    job_id = str(job.get("id") or "manual")

    results: list[dict] = []
    failed: list[dict] = []

    for index, item in enumerate(items, start=1):
        source_label = item["source"]
        try:
            print(f"[{index}/{len(items)}] Transcribing {source_label} (model={model_id})", flush=True)

            data = fetch_item_bytes(item)

            transcription = transcribe_bytes(
                data=data,
                label=source_label,
                model=model,
                language=language,
                vad_filter=vad_filter,
                apply_denoise=apply_denoise,
                beam_size=beam_size,
                initial_prompt=initial_prompt,
                include_segments=include_segments,
            )

            output_key = None
            if save_output:
                stem = Path(source_label).stem or source_label
                output_key = write_output(job_id, stem, transcription["text"])

        except Exception as exc:  # keep the rest of the batch going
            print(f"Failed to transcribe {source_label}: {exc}", flush=True)
            failed.append({"source": source_label, "error": str(exc)})
            continue

        results.append({
            "source": source_label,
            "output_key": output_key,
            "model": model_id,
            **transcription,
        })

    if not results:
        details = "; ".join(f"{item['source']}: {item['error']}" for item in failed)
        raise RuntimeError(f"No sources could be transcribed ({details})")

    response = {
        "results": results,
        "count": len(results),
        "failed": failed,
    }

    # Keep a flat single-item response shape for the common single-source case.
    if len(items) == 1 and not failed:
        response.update(results[0])

    return response


# ---------------------------------------------------------------------------
# Local debug mode / Serverless mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if os.environ.get("LOCAL_TEST") == "1":
        import glob

        samples_dir = os.environ.get("LOCAL_TEST_DIR", "samples")
        wav_paths = sorted(glob.glob(os.path.join(samples_dir, "*.wav")))

        print(f"Running handler in LOCAL_TEST mode against {len(wav_paths)} file(s) in {samples_dir}", flush=True)

        test_job = {
            "id": "local-test",
            "input": {
                "sources": wav_paths,
                "save_output": False,
            },
        }

        result = handler(test_job)

        # Note: the .txt files next to each sample are the OLD/current
        # method's output, not verified ground truth, so this is a
        # side-by-side comparison, not an accuracy score. A generic WER
        # against them is meaningless when the baseline itself is wrong.
        for item in result["results"]:
            source = item["source"]
            old_path = os.path.splitext(source)[0] + ".txt"
            print("=" * 100)
            print(os.path.basename(source))
            print(f"  NEW (this worker) : {item['text']}")
            if os.path.exists(old_path):
                old_text = open(old_path).read().strip()
                print(f"  OLD (current method): {old_text}")

        if result["failed"]:
            print("Failed:", result["failed"])
    else:
        print("Starting Runpod Serverless worker", flush=True)
        runpod.serverless.start({"handler": handler})
