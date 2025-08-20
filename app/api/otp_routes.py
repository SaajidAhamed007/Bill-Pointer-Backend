from fastapi import APIRouter, HTTPException, Body
import requests

router = APIRouter()

API_KEY = "2e3accab-7d90-11f0-a562-0200cd936042"  # Replace with your 2Factor.in API key
otp_sessions = {}  # Store session IDs temporarily


# Send OTP
@router.post("/send-otp")
def send_otp_route(mobile_no: str = Body(..., embed=True)):
    try:
        url = f"https://2factor.in/API/V1/{API_KEY}/SMS/{mobile_no}/AUTOGEN"
        response = requests.get(url)
        data = response.json()

        if data['Status'] != 'Success':
            raise Exception(f"Failed to send OTP: {data.get('Details')}")

        otp_sessions[mobile_no] = data['Details']  # Save session ID
        return {"message": "OTP sent successfully", "session_id": data['Details']}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Verify OTP
@router.post("/verify-otp")
def verify_otp_route(mobile_no: str = Body(...), otp: str = Body(...)):
    try:
        session_id = otp_sessions.get(mobile_no)
        if not session_id:
            raise HTTPException(status_code=400, detail="No OTP session found. Please send OTP first.")

        url = f"https://2factor.in/API/V1/{API_KEY}/SMS/{mobile_no}/AUTOGEN"
        response = requests.get(url)
        data = response.json()

        if data['Status'] != 'Success':
            raise HTTPException(status_code=400, detail="Invalid OTP")

        otp_sessions.pop(mobile_no)
        return {"message": "OTP verified successfully"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
