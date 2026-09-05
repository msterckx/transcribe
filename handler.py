#!/usr/bin/env python3
"""
Runpod Serverless noisy-audio transcription worker.

Built for very noisy short-form recordings (VHF radio / ATC-style traffic).
Generic Whisper hallucinates badly on this kind of audio (wrong language,
invented words). The default model here is a Whisper large-v3 checkpoint
fine-tuned on the ATCO2 corpus of real air-traffic-control radio audio,
which is dramatically more accurate on this domain. See README.md for the
comparison against generic Whisper that justified this choice.

Expected Network Volume mount:
    /runpod-volume

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

The worker reads:
    /runpod-volume/<source>

and, unless "save_output" is false, writes a plain-text transcript to:
    /runpod-volume/transcribe/output/<job-id>/<filename>.txt

Results are always returned in the "results" list, in the same order as the
requested sources; sources that fail are reported in "failed" while the rest
of the batch still completes.
"""

from __future__ import annotations

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
DATA_ROOT = VOLUME_ROOT / "transcribe"
OUTPUT_DIR = DATA_ROOT / "output"

HF_HOME = DATA_ROOT / "cache" / "huggingface"

if LOCAL_TEST:
    # /runpod-volume doesn't exist on a laptop; resolve sources relative to
    # the current directory instead and leave the HF cache at its default
    # location rather than trying to create it under a missing volume root.
    pass
else:
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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

def load_audio(path: Path) -> tuple[np.ndarray, float]:
    """
    Decode any ffmpeg-readable audio file to mono float32 PCM at 16kHz,
    matching Whisper's expected input regardless of source format/rate.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", "16000",
        "-loglevel", "error", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path.name}: {proc.stderr.decode(errors='replace')}")

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

def resolve_source(source_key: str) -> Path:
    """
    Resolve a relative Network Volume key safely beneath /runpod-volume.
    In LOCAL_TEST mode, resolve beneath the current directory instead, so
    the handler can run directly against ./samples without a real volume.
    """
    root = Path.cwd() if LOCAL_TEST else VOLUME_ROOT

    source_key = source_key.lstrip("/")
    source_path = (root / source_key).resolve()
    volume_root = root.resolve()

    try:
        source_path.relative_to(volume_root)
    except ValueError as exc:
        raise ValueError("source must stay inside the Network Volume") from exc

    if not source_path.is_file():
        raise FileNotFoundError(f"Source audio not found: {source_path}")

    if source_path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        raise ValueError(
            f"Unsupported source extension: {source_path.suffix}. "
            f"Supported: {sorted(SUPPORTED_INPUT_EXTENSIONS)}"
        )

    return source_path


def collect_source_keys(job_input: dict) -> list[str]:
    """
    Accept "source"/"sources" as either a single string or a list of strings
    and normalise them into an ordered, de-duplicated list of keys.
    """
    raw_values = []

    for field in ("source", "sources"):
        value = job_input.get(field)

        if value is None:
            continue
        if isinstance(value, str):
            raw_values.append(value)
        elif isinstance(value, (list, tuple)):
            raw_values.extend(value)
        else:
            raise ValueError(f'"{field}" must be a string or a list of strings')

    source_keys = []

    for value in raw_values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("every source must be a non-empty string")

        key = value.strip()
        if key not in source_keys:
            source_keys.append(key)

    if not source_keys:
        raise ValueError(
            'Missing required input field "source", for example '
            '"transcribe/input/clip.wav" or ["transcribe/input/a.wav", '
            '"transcribe/input/b.wav"]'
        )

    return source_keys


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_one(
    source_path: Path,
    model: WhisperModel,
    language: str | None,
    vad_filter: bool,
    apply_denoise: bool,
    beam_size: int,
    initial_prompt: str | None,
    include_segments: bool,
) -> dict:
    audio, duration = load_audio(source_path)

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

    source_keys = collect_source_keys(job_input)

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
    job_output_dir = OUTPUT_DIR / job_id

    results: list[dict] = []
    failed: list[dict] = []

    for index, source_key in enumerate(source_keys, start=1):
        try:
            source_path = resolve_source(source_key)

            print(f"[{index}/{len(source_keys)}] Transcribing {source_path} (model={model_id})", flush=True)

            transcription = transcribe_one(
                source_path=source_path,
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
                job_output_dir.mkdir(parents=True, exist_ok=True)
                txt_path = job_output_dir / f"{source_path.stem}.txt"
                txt_path.write_text(transcription["text"] + "\n")
                output_key = txt_path.relative_to(VOLUME_ROOT).as_posix()

        except Exception as exc:  # keep the rest of the batch going
            print(f"Failed to transcribe {source_key}: {exc}", flush=True)
            failed.append({"source": source_key, "error": str(exc)})
            continue

        results.append({
            "source": source_key,
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
    if len(source_keys) == 1 and not failed:
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
