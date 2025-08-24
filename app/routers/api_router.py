from fastapi import APIRouter
from app.api import user_routes, billboard_routes

api_router = APIRouter()

api_router.include_router(user_routes.router, prefix="/users", tags=["Users"])
api_router.include_router(billboard_routes.router, prefix="/billboard", tags=["Billboard"])