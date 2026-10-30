"""
Konkani speech-to-text backend.

Endpoint expected by the Java client:
    POST /transcribe   (multipart/form-data, field name "audio", a WAV file)
    -> {"devanagari": "...", "roman": "..."}

Run:
    uvicorn main:app --host 127.0.0.1 --port 8000
"""
import io
import os
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from fastapi import FastAPI, File, HTTPException, UploadFile
from transformers import AutoModel

MODEL_ID = os.getenv("KONKANI_MODEL", "ai4bharat/indic-conformer-600m-multilingual")
LANG_CODE = "kok"  # Konkani
DECODER = os.getenv("KONKANI_DECODER", "ctc")  # "ctc" (faster) or "rnnt" (often more accurate)
TARGET_SR = 16000

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading model {MODEL_ID} ... (first run downloads it)")
    # Newer onnxruntime refuses weights that are symlinks pointing outside the model folder
    # (which is how the Hugging Face cache stores files). So: download into a normal folder,
    # turn any symlinks there into real files, and make the model's own loader use that folder.
    import shutil
    import huggingface_hub

    local_dir = os.getenv(
        "KONKANI_MODEL_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_files"),
    )
    huggingface_hub.snapshot_download(MODEL_ID, local_dir=local_dir)
    for root, _dirs, files in os.walk(local_dir):
        for name in files:
            path = os.path.join(root, name)
            if os.path.islink(path):
                target = os.path.realpath(path)
                os.unlink(path)
                shutil.copy2(target, path)

    # The model's custom code calls snapshot_download(repo_id=...) itself; point it at our folder.
    _orig_snapshot = huggingface_hub.snapshot_download

    def _local_snapshot(repo_id=None, *args, **kwargs):
        if repo_id == MODEL_ID:
            return local_dir
        return _orig_snapshot(repo_id, *args, **kwargs)

    huggingface_hub.snapshot_download = _local_snapshot
    state["model"] = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("Model loaded.")
    yield
    state.clear()


app = FastAPI(title="Konkani STT", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Devanagari -> Roman transliteration (rule table, no external library)
# ---------------------------------------------------------------------------
VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ru", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
}
MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ru",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e",
}
CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
}
VIRAMA = "्"
NUKTA = "़"
SIGNS = {"ं": "n", "ँ": "n", "ः": "h"}
DIGITS = {c: str(i) for i, c in enumerate("०१२३४५६७८९")}


def _is_letter(ch: str) -> bool:
    return ch in VOWELS or ch in CONSONANTS or ch in MATRAS or ch in SIGNS or ch == VIRAMA


def _transliterate_word(word: str) -> str:
    out = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if ch in CONSONANTS:
            base = CONSONANTS[ch]
            i += 1
            if i < n and word[i] == NUKTA:
                i += 1
            if i < n and word[i] == VIRAMA:        # half consonant: no vowel
                out.append(base)
                i += 1
            elif i < n and word[i] in MATRAS:      # explicit vowel sign
                out.append(base + MATRAS[word[i]])
                i += 1
            else:                                  # inherent 'a'
                at_word_end = i >= n or not _is_letter(word[i]) or word[i] in SIGNS
                # Drop the final schwa (kamal, not kamala), keep it before anusvara.
                if at_word_end and not (i < n and word[i] in SIGNS):
                    out.append(base)
                else:
                    out.append(base + "a")
        elif ch in VOWELS:
            out.append(VOWELS[ch])
            i += 1
        elif ch in SIGNS:
            out.append(SIGNS[ch])
            i += 1
        elif ch in DIGITS:
            out.append(DIGITS[ch])
            i += 1
        elif ch in (VIRAMA, NUKTA):
            i += 1
        else:
            out.append(ch)  # punctuation, spaces, latin letters pass through
            i += 1
    return "".join(out)


def to_roman(text: str) -> str:
    # Split on spaces so schwa-deletion is applied per word.
    return " ".join(_transliterate_word(w) for w in text.split(" "))


# ---------------------------------------------------------------------------
# Audio handling
# ---------------------------------------------------------------------------
def load_audio_16k_mono(raw: bytes) -> torch.Tensor:
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read audio (need WAV): {e}")
    mono = data.mean(axis=1)                       # (samples,)
    wav = torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0)  # (1, samples)
    if sr != TARGET_SR:
        wav = AF.resample(wav, sr, TARGET_SR)
    return wav


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    raw = await audio.read()
    if len(raw) < 1000:
        raise HTTPException(status_code=400, detail="Audio too short")

    wav = load_audio_16k_mono(raw)

    with torch.inference_mode():
        result = state["model"](wav, LANG_CODE, DECODER)

    if isinstance(result, (list, tuple)):
        result = result[0] if result else ""
    devanagari = str(result).strip()

    return {"devanagari": devanagari, "roman": to_roman(devanagari)}
