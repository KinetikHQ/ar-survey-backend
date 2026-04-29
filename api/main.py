"""FastAPI application entry point."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from config import settings
from models.database import Base
from models.session import engine

logger = logging.getLogger(__name__)

if settings.ENVIRONMENT.lower() == "development":
    # Dev convenience only. Production must use Alembic migrations.
    Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AR Survey & Inspection API",
    version="1.0.0",
    description="Backend for construction-site PPE detection from video clips.",
)

# ---------------------------------------------------------------------------
# CORS — restrict origins in production
# ---------------------------------------------------------------------------
_allowed_origins = settings.allowed_origins_list

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log full validation errors so we can see exactly what the client sent."""
    body = await request.body()
    logger.error(
        "422 Validation error on %s %s\n"
        "  Headers: %s\n"
        "  Raw body: %s\n"
        "  Errors: %s",
        request.method,
        request.url.path,
        {k: v for k, v in dict(request.headers).items() if k.lower() != "authorization"},
        body.decode("utf-8", errors="replace"),
        exc.errors(),
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


app.include_router(router)
