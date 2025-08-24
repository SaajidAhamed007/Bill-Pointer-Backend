"""
How to convert image to base64 and send POST request
"""

import requests
import base64
import json

def convert_image_to_base64(image_path):
    """Convert image file to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            # Read the image file as bytes
            image_bytes = image_file.read()
            
            # Convert to base64 string
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            
            print(f"✅ Image converted to base64")
            print(f"📊 Original size: {len(image_bytes)} bytes")
            print(f"📊 Base64 size: {len(base64_string)} characters")
            print(f"📋 First 50 characters: {base64_string[:50]}...")
            
            return base64_string
    except FileNotFoundError:
        print(f"❌ Error: Image file '{image_path}' not found")
        return None
    except Exception as e:
        print(f"❌ Error converting image: {e}")
        return None

def send_billboard_detection_request(image_base64):
    """Send billboard detection request with image"""
    
    url = "http://127.0.0.1:8000/api/billboards/detections/"
    
    # Create request payload
    payload = {
        "user_id": 1,
        "latitude": 28.6139,
        "longitude": 77.2090,
        "dimension": "10x5",
        "quality_score": 0.8,
        "image_bytes": image_base64  # This is the base64 image string
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("🚀 Sending POST request...")
        response = requests.post(url, json=payload, headers=headers)
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS! Billboard detection completed")
            result = response.json()
            
            # Print key results
            print(f"🆔 Detection ID: {result.get('id')}")
            print(f"📍 Status: {result.get('status')}")
            print(f"💬 Message: {result.get('message')}")
            
            billboard_content = result.get('billboard_content', {})
            print(f"📝 Extracted Text: {billboard_content.get('extracted_text', 'None')}")
            print(f"🛡️ Content Safe: {billboard_content.get('content_appropriate', 'Unknown')}")
            
        else:
            print("❌ ERROR in request")
            try:
                error_detail = response.json()
                print(f"Error: {error_detail}")
            except:
                print(f"Error text: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure server is running on http://127.0.0.1:8000")
    except Exception as e:
        print(f"❌ Request Error: {e}")

# Example usage
if __name__ == "__main__":
    print("🎯 Billboard Detection - Image Base64 Example")
    print("=" * 50)
    
    # Option 1: Use an existing image file
    image_file = "sample_billboard.jpg"  # Replace with your image path
    
    # Option 2: Create a simple test image if no file exists
    try:
        from PIL import Image
        import io
        
        # Create a test image with text
        img = Image.new('RGB', (200, 100), color='white')
        
        # Save as bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Convert to base64
        image_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        print("✅ Created test image and converted to base64")
        
    except ImportError:
        # Fallback: Use a minimal valid base64 image
        print("📝 Using minimal test image...")
        image_base64 = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/gA=="
    
    # Send the request
    send_billboard_detection_request(image_base64)
    
    print("\n" + "=" * 50)
    print("💡 To use your own image:")
    print("1. Put your image file in the project folder")
    print("2. Update 'image_file' variable with your image name")
    print("3. Run this script again")
