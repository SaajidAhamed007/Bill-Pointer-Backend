# Billboard Detection API - Sample curl requests

# Basic curl request with sample data
curl -X POST "http://127.0.0.1:8000/api/billboards/detections/" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "latitude": 28.6139,
    "longitude": 77.2090,
    "dimension": "10x5",
    "quality_score": 0.8,
    "image_bytes": "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH4QIVCwQIRPWK9wAAABl0RVh0Q29tbWVudABDcmVhdGVkIHdpdGggR0lNUFeBDhcAAAASSURBVDjLY2AYBaNgFIyCoQcAAAQQAAF/TXiOAAAAAElFTkSuQmCC"
  }'

# Get format information
curl -X GET "http://127.0.0.1:8000/api/billboards/format"

# Get all detections
curl -X GET "http://127.0.0.1:8000/api/billboards/get-detections/"

# Get user detections
curl -X GET "http://127.0.0.1:8000/api/billboards/detections/user/1"
