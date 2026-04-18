"""Simple Bearer-token authentication dependency."""

import secrets
import time
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt, JWTError
from config import settings

_bearer_scheme = HTTPBearer()

# For MVP/Dev, we'll keep the static API_KEY check but allow for JWT expansion.
# In production, we'd replace this with a proper OAuth2/OIDC flow.

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = time.time() + 3600  # 1 hour
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm="HS256")

def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> dict:
    """Validate the token. In dev, matches API_KEY. In prod, validates JWT."""
    token = credentials.credentials

    # 1. Check if it's the static dev key (Legacy support for MVP)
    if token == settings.API_KEY:
        return {"sub": "dev-user", "role": "admin"}

    # 2. Otherwise, treat as JWT
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
