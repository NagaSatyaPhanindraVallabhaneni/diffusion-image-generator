"""FastAPI service: generate digit images with the trained diffusion model."""

import os

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .sample import generate_images, grid_png_bytes, load_checkpoint

CHECKPOINT_PATH = os.environ.get("DIFFUSION_CHECKPOINT", "models/ema.pt")

TRAIN_HINT = (
    "No trained checkpoint was found. Train the model first, then restart the API:\n"
    "  python scripts/train.py --conditional --epochs 10\n"
    "  uvicorn diffusion.app:app --host 0.0.0.0 --port 8000"
)

app = FastAPI(
    title="Diffusion Image Generator",
    description="DDPM image generation (implemented from scratch) served over HTTP.",
    version="0.1.0",
)

_model_cache = None


def get_model():
    """Lazily load the EMA checkpoint; raise a clear error if it is missing."""
    global _model_cache
    if _model_cache is None:
        try:
            _model_cache = load_checkpoint(CHECKPOINT_PATH, device="cpu")
        except FileNotFoundError as exc:
            raise RuntimeError(str(exc)) from exc
    return _model_cache


class GenerateRequest(BaseModel):
    n_images: int = Field(default=4, ge=1, le=16, description="Images to generate (1-16)")
    digit: int | None = Field(
        default=None, ge=0, le=9, description="Digit to generate (conditional models only)"
    )
    seed: int = Field(default=0, description="Random seed for reproducibility")


@app.get("/health")
def health():
    trained = os.path.exists(CHECKPOINT_PATH)
    return {"status": "ok", "model_loaded": trained, "checkpoint": CHECKPOINT_PATH}


@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        model, diffusion, config = get_model()
    except RuntimeError:
        raise HTTPException(status_code=503, detail=TRAIN_HINT)
    if req.digit is not None and not config.get("conditional", False):
        raise HTTPException(
            status_code=400,
            detail="Model trained unconditionally; omit 'digit' or retrain with --conditional.",
        )
    try:
        images = generate_images(
            model, diffusion, req.n_images, digit=req.digit, seed=req.seed, device="cpu"
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    png = grid_png_bytes(images)
    return Response(content=png, media_type="image/png")
