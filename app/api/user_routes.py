from fastapi import APIRouter, HTTPException, Body, Depends, Request
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from app.core.jwt import create_access_token, decode_access_token
import random
from app.core.auth import jwt_required

SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    mobile_no = Column(String, unique=True, index=True)
    aadhar_no = Column(String, unique=True)

Base.metadata.create_all(bind=engine)

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

otp_store = {}  # {mobile_no: otp}


def generate_and_store_otp(mobile_no: str) -> str:
    otp = str(random.randint(100000, 999999))
    otp_store[mobile_no] = otp
    print(f"✅ OTP for {mobile_no}: {otp}")
    return otp




def verify_otp(mobile_no: str, otp: str) -> bool:
    return otp_store.get(mobile_no) == otp

@router.post("/signup/send-otp")
def signup_send_otp(user: dict = Body(...), db: Session = Depends(get_db)):
    mobile_no = user.get("mobile_no")
    if not mobile_no:
        raise HTTPException(status_code=400, detail="mobile_no is required")

    existing = db.query(User).filter(User.mobile_no == mobile_no).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    generate_and_store_otp(mobile_no)
    return {"message": "OTP generated (check backend console)", "mobile_no": mobile_no}


@router.post("/signup/verify-otp")
def signup_verify_otp(payload: dict = Body(...), db: Session = Depends(get_db)):
    mobile_no = payload.get("mobile_no")
    otp = payload.get("otp")

    if not mobile_no or not otp:
        raise HTTPException(status_code=400, detail="mobile_no and otp required")

    if not verify_otp(mobile_no, otp):
        raise HTTPException(status_code=400, detail="Invalid OTP")

    user_data = payload
    new_user = User(
        name=user_data["name"],
        age=user_data["age"],
        gender=user_data["gender"],
        mobile_no=user_data["mobile_no"],
        aadhar_no=user_data["aadhar_no"]
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token({"sub": new_user.mobile_no})

    return {
        "message": "Signup successful",
        "user": {"id": new_user.id, "name": new_user.name, "mobile_no": new_user.mobile_no},
        "access_token": token,
        "token_type": "bearer"
    }

# ----------------- Login -----------------
@router.post("/login/send-otp")
def login_send_otp(mobile_no: str = Body(..., embed=True), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.mobile_no == mobile_no).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    generate_and_store_otp(mobile_no)
    return {"message": "OTP generated (check backend console)"}


@router.post("/login/verify-otp")
def login_verify_otp(mobile_no: str = Body(...), otp: str = Body(...), db: Session = Depends(get_db)):
    if not verify_otp(mobile_no, otp):
        raise HTTPException(status_code=400, detail="Invalid OTP")

    user = db.query(User).filter(User.mobile_no == mobile_no).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    token = create_access_token({"sub": user.mobile_no, "id": user.id})
    return {
        "message": "Login successful",
        "user": {"id": user.id, "name": user.name, "mobile_no": user.mobile_no},
        "access_token": token,
        "token_type": "bearer"
    }

@router.get("/get-users")
def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return {"users": users}

from fastapi import Depends

@router.get("/get-user/{user_id}")
def get_user(
    user_id: int, 
    db: Session = Depends(get_db), 
    current_user: dict = Depends(jwt_required)
):
    print("Current User Payload:", current_user)

    if user_id != current_user.get("id"):
        raise HTTPException(status_code=403, detail="Not authorized to access this user")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"user": {"id": user.id, "name": user.name, "mobile_no": user.mobile_no}}


@router.post("/add-user")
def add_user(user: dict = Body(...), db: Session = Depends(get_db)):
    new_user = User(
        name=user["name"],
        age=user["age"],
        gender=user["gender"],
        mobile_no=user["mobile_no"],
        aadhar_no=user["aadhar_no"]
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully", "user": user}

