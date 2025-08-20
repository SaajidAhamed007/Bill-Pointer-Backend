from sqlalchemy import Column, Integer, String
from app.db.sqlite import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    mobile_no = Column(String, unique=True, index=True)
    name = Column(String)
    gender = Column(String)
    age = Column(Integer)
    aadhar_no = Column(String,unique=True, index=True)
