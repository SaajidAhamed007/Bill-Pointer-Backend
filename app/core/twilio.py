import requests

API_KEY = "2e3accab-7d90-11f0-a562-0200cd936042"  # Get it from https://2factor.in

def send_otp_2factor(mobile_no: str):
    """
    Send OTP to an Indian mobile number using 2Factor.in
    Returns the session ID for verification.
    """
    url = f"https://2factor.in/API/V1/{API_KEY}/SMS/{mobile_no}/AUTOGEN"
    response = requests.get(url)
    data = response.json()
    
    if data.get('Status') != 'Success':
        raise Exception(f"Failed to send OTP: {data.get('Details')}")
    
    return data.get('Details')  # This is the session ID


def verify_otp_2factor(session_id: str, otp: str):
    """
    Verify OTP using the session ID returned by send_otp_2factor
    Returns True if OTP is correct.
    """
    url = f"https://2factor.in/API/V1/{API_KEY}/SMS/VERIFY/{session_id}/{otp}"
    response = requests.get(url)
    data = response.json()
    
    if data.get('Status') != 'Success':
        raise Exception(f"OTP verification failed: {data.get('Details')}")
    
    return True

