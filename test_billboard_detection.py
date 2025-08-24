"""
Sample POST request for billboard detection endpoint
"""

import requests
import base64
import json

# Sample image - create a small test image or use existing one
def create_sample_base64_image():
    """Create a simple test image and convert to base64"""
    try:
        from PIL import Image
        import io
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
    except ImportError:
        # Fallback - use a minimal PNG base64 string
        return "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH4QIVCwQIRPWK9wAAABl0RVh0Q29tbWVudABDcmVhdGVkIHdpdGggR0lNUFeBDhcAAAASSURBVDjLY2AYBaNgFIyCoQcAAAQQAAF/TXiOAAAAAElFTkSuQmCC"

def test_billboard_detection():
    """Test the billboard detection endpoint"""
    
    # API endpoint
    url = "http://127.0.0.1:8000/api/billboards/detections/"
    
    # Sample request data
    request_data = {
        "user_id": 1,
        "latitude": 28.6139,
        "longitude": 77.2090,
        "dimension": "10x5",
        "quality_score": 0.8,
        "image_bytes": create_sample_base64_image()
    }
    
    # Headers
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("🚀 Sending POST request to billboard detection...")
        print(f"URL: {url}")
        print(f"Data: {json.dumps({k: v if k != 'image_bytes' else 'BASE64_IMAGE_DATA...' for k, v in request_data.items()}, indent=2)}")
        
        # Send POST request
        response = requests.post(url, json=request_data, headers=headers)
        
        print(f"\n📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Success! Billboard detection completed")
            result = response.json()
            print(f"Detection ID: {result.get('id')}")
            print(f"Status: {result.get('status')}")
            print(f"Message: {result.get('message')}")
            print(f"Billboard Content: {result.get('billboard_content', {}).get('extracted_text', 'No text')}")
        else:
            print("❌ Error in request")
            print(f"Error details: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the server is running on http://127.0.0.1:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_billboard_detection()
