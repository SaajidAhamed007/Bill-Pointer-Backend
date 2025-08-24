"""
Simple POST request to billboard detection route
"""

import requests
import json

def make_detection_request():
    """Make a POST request to the billboard detection endpoint"""
    
    # API endpoint
    url = "http://127.0.0.1:8000/api/billboards/detections/"
    
    # Sample data with minimal valid base64 image (1x1 pixel PNG)
    request_data = {
        "user_id": 1,
        "latitude": 28.6139,
        "longitude": 77.2090,
        "dimension": "10x5",
        "quality_score": 0.8,
        "image_bytes": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAGAWVMxygAAAABJRU5ErkJggg=="
    }
    
    # Headers
    headers = {
        "Content-Type": "application/json"
    }
    
    print("🚀 Making POST request to billboard detection...")
    print(f"URL: {url}")
    print(f"Request data: {json.dumps(request_data, indent=2)}")
    
    try:
        # Send POST request
        response = requests.post(url, json=request_data, headers=headers)
        
        print(f"\n📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS! Detection created")
            result = response.json()
            
            print(f"🆔 Detection ID: {result.get('id')}")
            print(f"📍 Status: {result.get('status')}")
            print(f"💬 Message: {result.get('message')}")
            
            # Show billboard content
            billboard_content = result.get('billboard_content', {})
            print(f"📝 Extracted Text: {billboard_content.get('extracted_text', 'None')}")
            print(f"🛡️ Content Safe: {billboard_content.get('content_appropriate', 'Unknown')}")
            
            return result
            
        else:
            print("❌ Error in request")
            try:
                error_detail = response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Error text: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Server not running on http://127.0.0.1:8000")
        print("💡 Make sure to start the server with: uvicorn main:app --reload")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("🎯 Billboard Detection POST Request Test")
    print("=" * 50)
    
    result = make_detection_request()
    
    if result:
        print("\n🎉 Request successful!")
        print("🌐 You can view all detections at: http://127.0.0.1:8000/api/billboards/get-detections/")
    else:
        print("\n❌ Request failed!")
