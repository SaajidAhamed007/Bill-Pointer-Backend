from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.sqlite import get_db
from app.models.detections import Detection
from datetime import datetime

router = APIRouter()

@router.post("/sync")
def sync_detections(payload: dict, db: Session = Depends(get_db)):
    last_pulled_at = payload.get("last_pulled_at", 0)
    changes_from_client = payload.get("changes", [])

    # 1. Push client changes → server
    for change in changes_from_client:
        detection = db.query(Detection).filter(Detection.id == change["id"]).first()
        if detection:
            detection.status = change.get("status", detection.status)
            detection.flagged_reason = change.get("flagged_reason", detection.flagged_reason)
            detection.quality_score = change.get("quality_score", detection.quality_score)
        else:
            new_det = Detection(**change)
            db.add(new_det)
    db.commit()

    # 2. Pull changes from server → client
    updated_detections = db.query(Detection).filter(
        Detection.updated_at > last_pulled_at
    ).all()

    return {
        "timestamp": int(datetime.utcnow().timestamp()),
        "changes": [ 
            {
                "id": d.id,
                "status": d.status.value,
                "flagged_reason": d.flagged_reason,
                "quality_score": d.quality_score,
                "content": d.content,
                "dimension": d.dimension,
                "latitude": d.latitude,
                "longitude": d.longitude,
                "image_path": d.image_path
            }
            for d in updated_detections
        ]
    }
