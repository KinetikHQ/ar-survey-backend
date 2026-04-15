"""SQLAlchemy session factory and engine."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import settings

# SQLite needs no connection pool; Postgres does.
_connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}
    engine = create_engine(settings.DATABASE_URL, connect_args=_connect_args)
else:
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
