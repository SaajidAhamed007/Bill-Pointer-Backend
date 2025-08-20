from fastapi import APIRouter, HTTPException, Form, UploadFile, BackgroundTasks, status, Depends, Request
from app.core.jwt import create_access_token, decode_access_token

def jwt_required(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing or invalid"
        )

    token = auth_header.split(" ")[1]
    payload = decode_access_token(token)
    if not payload or "id" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    return payload