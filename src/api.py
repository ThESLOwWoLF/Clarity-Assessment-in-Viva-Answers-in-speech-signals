"""FastAPI service + Gradio demo.

    uvicorn src.api:app --port 8000        # POST /predict, demo at http://localhost:8000/demo
Env: MODEL_PATH (default from config.yaml), DEVICE (cpu/cuda, default auto).
"""
import io
import os
import time
from contextlib import asynccontextmanager

import gradio as gr
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile

from src.data import load_config
from src.features import SR, load_audio
from src.predict import Predictor

cfg = load_config()
MAX_S = cfg["api_max_seconds"]
MIN_SR, MAX_SR = 8000, 48000
state = {}


@asynccontextmanager
async def lifespan(app):
    state["p"] = Predictor(os.environ.get("MODEL_PATH"), os.environ.get("DEVICE"))
    yield


app = FastAPI(title="Viva answer clarity", lifespan=lifespan)


def run(data: bytes):
    t0 = time.perf_counter()
    try:
        info = sf.info(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Not a readable WAV file.")
    if info.format != "WAV":
        raise HTTPException(400, f"Expected WAV, got {info.format}.")
    if not MIN_SR <= info.samplerate <= MAX_SR:
        raise HTTPException(400, f"Sample rate {info.samplerate} Hz not in [{MIN_SR}, {MAX_SR}].")
    if info.duration > MAX_S:
        raise HTTPException(413, f"Clip is {info.duration:.1f} s; max is {MAX_S} s.")
    if info.duration < 0.5:
        raise HTTPException(400, "Clip shorter than 0.5 s.")
    wav, _ = load_audio(io.BytesIO(data))  # stereo -> mono, resample to 16 kHz
    label, probs = state["p"].predict_wav(wav)
    return {"label": label, "probabilities": probs, "duration_s": round(len(wav) / SR, 3),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/predict")
async def predict(file: UploadFile):
    data = await file.read(int(MAX_S * MAX_SR * 2 * 4) + 1024)  # cap read size: max-length stereo float32
    return run(data)


@app.get("/health")
def health():
    return {"ok": "p" in state, "device": str(state["p"].embed.device) if "p" in state else None}


def _demo(path):
    if path is None:
        return "Upload or record a clip.", None
    with open(path, "rb") as f:
        try:
            r = run(f.read())
        except HTTPException as e:
            return f"Error: {e.detail}", None
    return f"{r['label']}  ({r['duration_s']} s, {r['latency_ms']} ms)", r["probabilities"]


demo = gr.Interface(_demo, gr.Audio(sources=["upload", "microphone"], type="filepath", format="wav"),
                    [gr.Textbox(label="Clarity"), gr.Label(label="Probabilities")],
                    title="Viva answer clarity", flagging_mode="never")
app = gr.mount_gradio_app(app, demo, path="/demo")
