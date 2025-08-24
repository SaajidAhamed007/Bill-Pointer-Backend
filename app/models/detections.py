from sqlalchemy import Column, Integer, String, Float, Enum as SqlEnum
from app.db.sqlite import Base
import enum

class DetectionStatus(enum.Enum):
    pending = "pending"
    approved = "approved"
    flagged = "flagged"

class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    image_path = Column(String, nullable=False)
    dimension = Column(String, nullable=False)
    content = Column(String, nullable=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    quality_score = Column(Float, nullable=False)
    status = Column(SqlEnum(DetectionStatus), default=DetectionStatus.pending, nullable=False)
    flagged_reason = Column(String, nullable=True)
    updated_at = Column(Integer, nullable=False, index=True)