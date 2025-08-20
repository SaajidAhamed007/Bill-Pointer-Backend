from fastapi import APIRouter, UploadFile, Form, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.sqlite import get_db
from app.models.detections import Detection, DetectionStatus
import shutil
import os
import uuid
import requests

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
    user_id: int = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    dimension: str = Form(...),
    quality_score: float = Form(...),
    image: UploadFile = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    # Save image
    file_ext = os.path.splitext(image.filename)[1]
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