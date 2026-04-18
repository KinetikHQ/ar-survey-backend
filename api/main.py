"""FastAPI application entry point."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.routes import router
from models.database import Base
from models.session import engine

logger = logging.getLogger(__name__)

# Create tables on startup (fine for dev; use Alembic in production)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AR Survey & Inspection API",
    version="1.0.0",
    description="Backend for construction-site PPE detection from video clips.",
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
