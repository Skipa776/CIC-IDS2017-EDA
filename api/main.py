"""
CICIDS2017 Intrusion Detection System API

FastAPI application for real-time network flow classification.
"""

import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from api.config import settings
from api.models.classifier import classifier
from api.routers import classify_router, health_router


# Configure logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        return json.dumps(log_entry)


logger = logging.getLogger("ids_api")
logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)
handler = logging.StreamHandler()
if not settings.debug:
    handler.setFormatter(JSONFormatter())
else:
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(handler)


# Request ID middleware
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Loads models on startup and cleans up on shutdown.
    """
    logger.info("Loading IDS models...")
    try:
        classifier.load_models()
        logger.info(
            "Models loaded successfully from %s (Layer1=%s, Layer2=%s, Features=%d, Classes=%d)",
            classifier.model_dir,
            type(classifier.layer1_model).__name__,
            type(classifier.layer2_model).__name__,
            len(classifier.feature_columns),
            len(classifier.label_mapping),
        )
    except Exception as e:
        logger.warning("Failed to load models: %s — API will start in degraded mode", e)

    yield

    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## CICIDS2017 Intrusion Detection System API

This API provides real-time network flow classification using a layered machine learning approach:

### Architecture
1. **Layer 1 (Binary)**: Fast classification to detect if traffic is an attack
2. **Layer 2 (Multi-class)**: Detailed attack type identification (15 classes)
3. **MITRE ATT&CK Mapping**: Attack techniques and mitigations

### Attack Types Detected
- DoS attacks (Hulk, GoldenEye, Slowhttptest, slowloris)
- DDoS attacks
- Brute Force (FTP-Patator, SSH-Patator, Web)
- Web Attacks (XSS, SQL Injection)
- Port Scanning
- Botnet traffic
- Infiltration
- Heartbleed exploit

### Performance
- Binary classification PR-AUC: >0.99
- Multi-class classification F1: >0.93
- Inference time: <1ms per flow
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add Request ID middleware
app.add_middleware(RequestIDMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(classify_router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
        "classify": f"{settings.api_prefix}/classify",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
