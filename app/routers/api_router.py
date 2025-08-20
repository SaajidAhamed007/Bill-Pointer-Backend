from fastapi import APIRouter
from app.api import user_routes, test, otp_routes

api_router = APIRouter()

api_router.include_router(user_routes.router, prefix="/users", tags=["Users"])
api_router.include_router(test.router, prefix="/test", tags=["Test"])
api_router.include_router(otp_routes.router, prefix="/otp", tags=["OTP"])