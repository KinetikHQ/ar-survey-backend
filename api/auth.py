"""Simple Bearer-token authentication dependency."""

import time
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt as pyjwt
from config import settings

_bearer_scheme = HTTPBearer()


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = time.time() + 3600  # 1 hour
    to_encode.update({"exp": expire})
    return pyjwt.encode(to_encode, settings.JWT_SECRET, algorithm="HS256")


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> dict:
    """Validate the token. In dev, matches API_KEY. In prod, validates JWT."""
    token = credentials.credentials

    # 1. Check if it's the static dev key (dev only)
    if token == settings.API_KEY and settings.ENVIRONMENT == "development":
        return {"sub": "dev-user", "role": "admin"}

    # 2. Otherwise, treat as JWT
    try:
        payload = pyjwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        return payload
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except pyjwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
