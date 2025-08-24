from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.sqlite import get_db
from app.models.detections import Detection, DetectionStatus
from app.core.auth import jwt_required
import shutil
import os
import uuid
import requests
import time
from typing import List, Optional

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---- Placeholder AI Functions ---- #
def is_billboard(image_path: str) -> bool:
    """
    AI model to check if uploaded image is a billboard.
    Return True if billboard, else False.
    """
    # TODO: integrate your detection model
    return True


def extract_text_from_image(image_path: str) -> str:
    """
    OCR function to extract text from image.
    """
    # TODO: integrate your OCR model
    return "Sample Billboard Text"


def check_if_flagged(content: str, dimension: str, latitude: float, longitude: float, quality_score: float):
    """
    AI model to decide if billboard should be flagged.
    Returns (status, flagged_reason)
    """
    # TODO: implement your logic/model
    # Example: if quality < 0.5 → flagged
    if quality_score < 0.5:
        return DetectionStatus.flagged, "Low quality"
    return DetectionStatus.approved, None


def send_alert_to_municipality(detection: Detection):
    """
    Send alert to municipality when a billboard is flagged.
    """
    municipality_api = "https://municipality.example.com/api/alerts"  # replace with real endpoint

    # Replace with your actual public image URL logic if needed
    public_image_url = f"https://yourapp.com/uploads/{os.path.basename(detection.image_path)}"

    payload = {
        "billboard_id": detection.id,
        "user_id": detection.user_id,
        "location": {"latitude": detection.latitude, "longitude": detection.longitude},
        "dimension": detection.dimension,
        "content": detection.content,
        "image_url": public_image_url,
        "flagged_reason": detection.flagged_reason,
        "quality_score": detection.quality_score
    }

    try:
        response = requests.post(municipality_api, json=payload, timeout=5)
        response.raise_for_status()
    except Exception as e:
        # log the error, but do not block the detection
        print(f"Failed to notify municipality: {e}")


# ---- Route ---- #
@router.post("/detections/")
async def create_detection(
    user_id: int = Form(..., description="User ID"),
    latitude: float = Form(..., description="Latitude coordinate"),
    longitude: float = Form(..., description="Longitude coordinate"),
    dimension: str = Form(..., description="Billboard dimensions"),
    quality_score: float = Form(..., description="Quality score (0.0 to 1.0)"),
    image: UploadFile = File(..., description="Billboard image file"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    # Validate image file
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate quality score range
    if not 0.0 <= quality_score <= 1.0:
        raise HTTPException(status_code=400, detail="Quality score must be between 0.0 and 1.0")
    
    # Save image
    file_ext = os.path.splitext(image.filename)[1]
    if not file_ext:
        file_ext = ".jpg"  # Default extension if none provided
    file_name = f"{uuid.uuid4().hex}{file_ext}"
    image_path = os.path.join(UPLOAD_DIR, file_name)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Step 1: Check billboard
    if not is_billboard(image_path):
        os.remove(image_path)
        raise HTTPException(status_code=400, detail="Uploaded image is not a billboard.")

    # Step 2: Extract text
    content = extract_text_from_image(image_path)

    # Step 3: AI flagging
    status, flagged_reason = check_if_flagged(content, dimension, latitude, longitude, quality_score)

    # Step 4: Save in DB
    detection = Detection(
        user_id=user_id,
        image_path=image_path,
        dimension=dimension,
        content=content,
        latitude=latitude,
        longitude=longitude,
        quality_score=quality_score,
        status=status,
        flagged_reason=flagged_reason,
        updated_at=int(time.time())
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)

    # Step 5: Send alert in background if flagged
    if status == DetectionStatus.flagged:
        background_tasks.add_task(send_alert_to_municipality, detection)

    return {
        "id": detection.id,
        "user_id": detection.user_id,
        "image_path": detection.image_path,
        "dimension": detection.dimension,
        "content": detection.content,
        "latitude": detection.latitude,
        "longitude": detection.longitude,
        "status": detection.status.value,
        "quality_score": detection.quality_score,
        "flagged_reason": detection.flagged_reason,
    }


@router.get("/detections/user/{user_id}")
async def get_user_detections(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(jwt_required),
    status: Optional[str] = None
):
    """
    Get all billboard detections for a specific user.
    Users can only access their own detections.
    """
    # Check if the current user is accessing their own data
    if user_id != current_user.get("id"):
        raise HTTPException(
            status_code=403, 
            detail="Not authorized to access detections for this user"
        )
    
    # Build query
    query = db.query(Detection).filter(Detection.user_id == user_id)
    
    # Filter by status if provided
    if status:
        try:
            status_enum = DetectionStatus(status)
            query = query.filter(Detection.status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid status. Valid options: {[s.value for s in DetectionStatus]}"
            )
    
    detections = query.all()
    
    return {
        "user_id": user_id,
        "total_detections": len(detections),
        "detections": [
            {
                "id": detection.id,
                "user_id": detection.user_id,
                "image_path": detection.image_path,
                "dimension": detection.dimension,
                "content": detection.content,
                "latitude": detection.latitude,
                "longitude": detection.longitude,
                "status": detection.status.value,
                "quality_score": detection.quality_score,
                "flagged_reason": detection.flagged_reason,
                "updated_at": detection.updated_at
            }
            for detection in detections
        ]
    }


@router.get("/detections/")
async def get_all_detections(
    db: Session = Depends(get_db),
    status: Optional[str] = None,
    limit: Optional[int] = 100,
    offset: Optional[int] = 0
):
    """
    Get all billboard detections (admin/public view).
    Supports filtering by status and pagination.
    """
    # Build query
    query = db.query(Detection)
    
    # Filter by status if provided
    if status:
        try:
            status_enum = DetectionStatus(status)
            query = query.filter(Detection.status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid status. Valid options: {[s.value for s in DetectionStatus]}"
            )
    
    # Apply pagination
    total_count = query.count()
    detections = query.offset(offset).limit(limit).all()
    
    return {
        "total_count": total_count,
        "returned_count": len(detections),
        "offset": offset,
        "limit": limit,
        "detections": [
            {
                "id": detection.id,
                "user_id": detection.user_id,
                "image_path": detection.image_path,
                "dimension": detection.dimension,
                "content": detection.content,
                "latitude": detection.latitude,
                "longitude": detection.longitude,
                "status": detection.status.value,
                "quality_score": detection.quality_score,
                "flagged_reason": detection.flagged_reason,
                "updated_at": detection.updated_at
            }
            for detection in detections
        ]
    }


@router.get("/detections/{detection_id}")
async def get_detection_by_id(
    detection_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(jwt_required)
):
    """
    Get a specific billboard detection by ID.
    Users can only access their own detections.
    """
    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    # Check if the current user owns this detection
    if detection.user_id != current_user.get("id"):
        raise HTTPException(
            status_code=403, 
            detail="Not authorized to access this detection"
        )
    
    return {
        "id": detection.id,
        "user_id": detection.user_id,
        "image_path": detection.image_path,
        "dimension": detection.dimension,
        "content": detection.content,
        "latitude": detection.latitude,
        "longitude": detection.longitude,
        "status": detection.status.value,
        "quality_score": detection.quality_score,
        "flagged_reason": detection.flagged_reason,
        "updated_at": detection.updated_at
    }