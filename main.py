from fastapi import FastAPI
from app.routers.api_router import api_router
from app.db.sqlite import Base, engine

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Your Project Name")

# Include API router
app.include_router(api_router, prefix="/api")
