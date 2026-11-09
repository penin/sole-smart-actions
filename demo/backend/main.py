"""Sole Smart Actions demo API.

    python demo/backend/main.py            # http://127.0.0.1:8000 (serves the built frontend if present)

The trained model is loaded once at startup, read-only. Threshold changes are per-request only and are
never written back to configuration. Request text is not logged.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import paths  # noqa: E402
from decision import apply_threshold  # noqa: E402
from model_service import ModelRegistry, ModelUnavailableError, build_registry  # noqa: E402
from schemas import (Decision, DecideRequest, ModelSummary, PredictRequest, PredictResponse,  # noqa: E402
                     RankedProbability)


def create_app(registry: ModelRegistry | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.registry = registry or build_registry()
        yield

    app = FastAPI(title="Sole Smart Actions demo API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    def get_model(request: Request, model_id: str | None):
        try:
            model = request.app.state.registry.get(model_id)
        except KeyError:
            raise HTTPException(404, f"Unknown model '{model_id}'")
        if not model.available:
            reason = getattr(model, "load_error", None) or f"{model.name} is not available yet"
            raise HTTPException(503, reason)
        return model

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        # Return field + message only; never echo the submitted text back into logs or errors.
        errors = [{"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": "Invalid request", "errors": errors})

    @app.exception_handler(ModelUnavailableError)
    async def unavailable_handler(request: Request, exc: ModelUnavailableError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": f"Unexpected server error ({type(exc).__name__})"})

    @app.get("/api/health")
    def health(request: Request):
        reg = request.app.state.registry
        default = reg.get(None)
        return {"status": "ok" if default.available else "degraded", "model_loaded": default.available,
                "model_id": default.id, "error": getattr(default, "load_error", None)}

    @app.get("/api/models", response_model=list[ModelSummary])
    def models(request: Request):
        return request.app.state.registry.summaries()

    @app.get("/api/model-info")
    def model_info(request: Request, model_id: str | None = None):
        return get_model(request, model_id).info()

    @app.post("/api/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request):
        model = get_model(request, body.model_id)
        default = model.default_threshold
        threshold = default if body.threshold is None else body.threshold
        raw = model.predict(body.text)
        decision = apply_threshold(raw.probabilities, threshold, raw.class_order)
        ranked = sorted(raw.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        return PredictResponse(
            **decision, model_id=model.id, model_name=model.name, input=body.text,
            class_order=raw.class_order, probabilities=raw.probabilities,
            ranked=[RankedProbability(intent=k, probability=v) for k, v in ranked],
            default_threshold=default, recognized_tokens=raw.explanation["recognized_tokens"],
            unknown_tokens=raw.explanation["unknown_tokens"], explanation=raw.explanation,
            inference_ms=raw.inference_ms,
        )

    @app.post("/api/decide", response_model=Decision)
    def decide(body: DecideRequest, request: Request):
        """Re-apply threshold + fallback rules to existing probabilities. No inference is run."""
        model = get_model(request, body.model_id)
        try:
            return apply_threshold(body.probabilities, body.threshold, model.class_order)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    # Presentation mode: serve the built frontend from the same process.
    if paths.FRONTEND_DIST.exists():
        app.mount("/assets", StaticFiles(directory=paths.FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str):
            candidate = (paths.FRONTEND_DIST / full_path).resolve()
            if full_path and candidate.is_file() and paths.FRONTEND_DIST.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(paths.FRONTEND_DIST / "index.html")

    return app


app = create_app()

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Sole Smart Actions demo API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
