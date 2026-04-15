"""FastAPI application entry point."""

from fastapi import FastAPI

from api.routes import router
from models.database import Base
from models.session import engine

# Create tables on startup (fine for dev; use Alembic in production)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AR Survey & Inspection API",
    version="1.0.0",
    description="Backend for construction-site PPE detection from video clips.",
)

app.include_router(router)
