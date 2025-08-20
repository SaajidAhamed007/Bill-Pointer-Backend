from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()



@router.get("/hello")
def hello():
    return {"message": "Hello, World!"}


