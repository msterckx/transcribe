# transcribe

Runpod Serverless worker for transcribing very noisy short-form audio (VHF
radio / ATC-style traffic) in batches. Same shape as [`image-upscale`](https://github.com/msterckx/image-upscale)
and [`voicestudio`](https://github.com/msterckx/voicestudio): a single
`handler.py`, a `Dockerfile.txt`, and a GitHub Actions workflow that builds
and pushes the image to Docker Hub on every change.

## Why not generic Whisper

The `samples/` folder in this repo (`EBAW_APP_*.wav` — Antwerp Approach
control) is real narrowband ATC radio audio, plus `.txt` files holding
whatever the previous method produced. Generic Whisper (any size) hallucinates
badly on this kind of audio: it flips languages mid-clip and invents fluent
nonsense instead of admitting it can't hear the words. For example, on
`EBAW_APP_20260905_074001_329.wav`, faster-whisper `small` with language
auto-detect produced:

```
[de 0.81] Wir sind 2040.
```

There is no German on this frequency. That kind of confident-but-wrong output
is exactly what "current methods are not working" looks like.

The fix used here is **not** generic denoising — a bandpass filter + spectral
noise gate was tested and made things worse (it clipped word endings and
sometimes dropped short transmissions entirely). What actually works is a
Whisper checkpoint fine-tuned directly on real noisy ATC radio audio:
[`jlvdoorn/whisper-large-v3-atco2-asr`](https://huggingface.co/jlvdoorn/whisper-large-v3-atco2-asr),
trained on the [ATCO2](https://www.atco2.org/) corpus. Same file, same raw
audio, this model:

```
descend two thousand feet
```

That's a real, plausible ATC instruction, and it's what this worker uses by
default. A few more side-by-sides from `samples/` (OLD = previous method's
`.txt`, NEW = this worker):

| file | OLD (current method) | NEW (this worker) |
|---|---|---|
| `..._082708_327.wav` | So, take the whole speed back to 200. | level six zero speed back to two zero ... |
| `..._083143_470.wav` | 2-5 right? | ...two five right |
| `..._083944_369.wav` | 3,000 CC line by Papa Croix. | turning three thousand CC ... Papa Chris |

The OLD column is itself unreliable (that's the whole problem), so don't read
these as a graded accuracy score — read them as "does the output sound like
real ATC phraseology or invented English/German/French." Run
[Local testing](#local-testing) below to see the full comparison on all 15
samples and judge for yourself.

Also disabled by default: VAD filtering. It's designed to drop silence, but
on these short, low-SNR clips it also drops real speech at the noise floor —
one sample transcribed to nothing at all with VAD on, and correctly
(`two five right`) with it off.

## Input

```json
{
  "input": {
    "source": "transcribe/input/clip.wav"
  }
}
```

Batch multiple files in one job with `sources`:

```json
{
  "input": {
    "sources": ["transcribe/input/a.wav", "transcribe/input/b.wav"]
  }
}
```

`source`/`sources` are object keys, resolved against whichever storage
backend is configured (see [Storage backend](#storage-backend) below). Any
format `ffmpeg` can decode is accepted (wav, mp3, m4a, flac, ogg, ...);
everything is resampled to mono 16kHz internally.

**Or send the audio inline**, with no upload step at all, via
`audio_base64` (`audios_base64` for a batch):

```json
{
  "input": {
    "audio_base64": "UklGRi4AAABXQVZFZm10IBAAAAABAAEA..."
  }
}
```

Accepts a plain base64 string, a `data:audio/wav;base64,...` URI, or
`{"data": "...", "name": "my-clip"}` to control the label used in the
response and output filename (defaults to `inline-1`, `inline-2`, ...).
`source`/`sources` and `audio_base64`/`audios_base64` can be mixed in one
batch request. Inline audio is capped at 50MB decoded — for anything
larger, upload it to the storage backend and use `source` instead.

Optional fields (defaults shown):

| field | default | notes |
|---|---|---|
| `model` | `"atc-large-v3"` | alias for the baked-in ATC model, or any other alias below, or an arbitrary faster-whisper-compatible HF repo id (downloaded on first use and cached on the volume) |
| `language` | `"en"` | pass `null` to auto-detect instead of forcing English |
| `vad_filter` | `false` | trims silence; risky on short/noisy clips, see above |
| `denoise` | `false` | bandpass + spectral noise gate; measured worse on the ATC model, may help on other checkpoints/noise profiles |
| `beam_size` | `5` | |
| `initial_prompt` | `null` | text to bias decoding, e.g. toward callsigns or vocabulary you expect |
| `segments` | `false` | include per-segment timestamps in the response |
| `save_output` | `true` | also write a `.txt` transcript to the volume |

Built-in model aliases: `atc-large-v3` (default, baked into the image),
`large-v3`, `medium`, `small` (plain multilingual Whisper, for comparison).

## Output

```json
{
  "text": "descend two thousand feet",
  "language": "en",
  "language_probability": 0.94,
  "duration_seconds": 2.76,
  "source": "transcribe/input/clip.wav",
  "output_key": "transcribe/output/<job-id>/clip.txt",
  "model": "atc-large-v3"
}
```

For a batch (`sources`) request, the same per-item objects come back in a
`results` list (input order preserved), alongside `count` and `failed`
(sources that errored, with the error message, while the rest of the batch
still completes) — mirroring the batch shape used by `image-upscale`.

## Storage backend

Auto-detected, in this priority order. Whichever is active, `source`/
`sources` are read from it and outputs are written back to it under
`transcribe/...` key prefixes:

**1. Network Volume mount** (preferred whenever present) — attach a Network
Volume to the endpoint; RunPod mounts it at `/runpod-volume` in every worker
automatically. This is plain local disk — no credentials, no configuration,
and faster than the S3 API, so it's used whenever it's there, even if
`RUNPOD_S3_*` vars are also set on the endpoint.

**2. S3-compatible bucket** (fallback, only used when no volume is mounted) —
for endpoints that deliberately skip attaching a Network Volume (attaching
one pins the endpoint to that volume's data center region, which can limit
GPU availability). Set these on the endpoint (RunPod Console → your endpoint
→ Environment Variables), not in `Dockerfile.txt`:

| variable | example |
|---|---|
| `RUNPOD_S3_BUCKET` | your Network Volume ID |
| `RUNPOD_S3_ENDPOINT` | `https://s3api-<region>.runpod.io` |
| `RUNPOD_S3_ACCESS_KEY_ID` | from RunPod Console → Settings → S3 API Keys |
| `RUNPOD_S3_SECRET_ACCESS_KEY` | ditto |
| `RUNPOD_S3_REGION` | e.g. `eu-ro-1` (optional, defaults to `us-east-1`) |

This account's `image-upscale` volume/bucket already has `upscale/...` keys
in it; this worker only ever reads/writes under `transcribe/...`, so the two
coexist safely in the same bucket whichever way it's accessed.

If your endpoint has a Network Volume attached, you don't need the
`RUNPOD_S3_*` variables at all — leave them unset.

## Local testing

No GPU or Runpod account needed to sanity-check the pipeline against
`samples/`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install faster-whisper huggingface_hub numpy scipy noisereduce runpod

LOCAL_TEST=1 FORCE_CPU=1 python3 handler.py
```

This transcribes every `.wav` in `samples/` on CPU and prints each result
next to the matching `.txt` (the old method's output), if present. First run
downloads the ~3GB ATC model from Hugging Face to `~/.cache/huggingface`;
subsequent runs reuse the cache.

## Deploying

1. Add `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` secrets to this repo (same as
   `image-upscale`/`voicestudio`) so `.github/workflows/docker-build.yml` can
   push `msterckx/transcribe-serverless`. Each build pushes three tags:
   a version (`:1.1`), the commit SHA (`:<short-sha>`), and `:latest`. Point
   the endpoint at the SHA tag if you want an exact, reproducible build
   instead of `:latest` — RunPod doesn't always re-pull `:latest` on an
   already-running worker, so a SHA tag is the reliable way to force a
   specific version.
2. In Runpod, create a Serverless Endpoint from that image, and either attach
   a Network Volume, or set the `RUNPOD_S3_*` environment variables (see
   [Storage backend](#storage-backend)) for S3 access instead.
3. Upload input audio under `transcribe/input/` on whichever backend you
   chose (matching the `source`/`sources` keys you send in job input).
4. GPU is recommended (`compute_type=float16`) but the handler falls back to
   CPU automatically if CUDA isn't available (`FORCE_CPU=1` forces this).

## Testing a live endpoint

With a file already on the storage backend:

```bash
curl -s -X POST "https://api.runpod.ai/v2/<endpoint-id>/runsync" \
  -H "Authorization: Bearer <runpod-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"input": {"source": "transcribe/input/clip.wav"}}' | python3 -m json.tool
```

Or send a local file inline, no upload step needed:

```bash
curl -s -X POST "https://api.runpod.ai/v2/<endpoint-id>/runsync" \
  -H "Authorization: Bearer <runpod-api-key>" \
  -H "Content-Type: application/json" \
  -d "{\"input\": {\"audio_base64\": \"$(base64 -i clip.wav)\"}}" | python3 -m json.tool
```

Note the input key is `source`/`sources`/`audio_base64`/`audios_base64`, not
`prompt` — this worker isn't a text-generation endpoint.
