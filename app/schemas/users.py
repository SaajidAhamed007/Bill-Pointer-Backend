from pydantic import BaseModel

class UserCreate(BaseModel):
    name: str
    age: int
    gender: str
    mobile_no: str
    aadhar_no: str

class UserOut(BaseModel):
    id: int
    name: str
    age: int
    gender: str
    mobile_no: str
    aadhar_no: str

    model_config = {
        "from_attributes": True
    }
