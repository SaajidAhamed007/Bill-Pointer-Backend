#!/usr/bin/env python3
"""
Simple POST request test for Billboard Detection API
"""

import requests
import json
import base64
from io import BytesIO
from PIL import Image
import time

def create_sample_image():
    """Create a simple test image as base64"""
    # Create a simple colored image
    img = Image.new('RGB', (400, 200), color='blue')
    
    # Add some text-like rectangles to simulate a billboard
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Draw a white rectangle (like a billboard background)
    draw.rectangle([50, 50, 350, 150], fill='white', outline='black', width=2)
    
    # Draw some colored rectangles (simulate text/logos)
    draw.rectangle([70, 70, 150, 90], fill='red')
    draw.rectangle([160, 70, 240, 90], fill='green')
    draw.rectangle([250, 70, 330, 90], fill='blue')
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64

def test_detection_endpoint():
    """Test the billboard detection endpoint"""
    
    # API endpoint
    url = "http://localhost:8000/api/billboards/detections/"
    
    # Create sample data
    sample_image_base64 = create_sample_image()
    
    # Test data for the POST request (corrected format)
    test_data = {
        "user_id": 1,                 # Required: User ID
        "latitude": 28.6139,          # Delhi coordinates
        "longitude": 77.2090,
        "dimension": "20x10",         # Required: Billboard dimensions
        "quality_score": 0.8,         # Required: Image quality (0.0-1.0)
        "image_bytes": sample_image_base64  # Required: Base64 image data
    }
    
    print("🚀 Testing Billboard Detection API")
    print(f"📍 Location: {test_data['latitude']}, {test_data['longitude']}")
    print(f"� Dimension: {test_data['dimension']}")
    print(f"👤 User ID: {test_data['user_id']}")
    print(f"⭐ Quality Score: {test_data['quality_score']}")
    print(f"📊 Image size: {len(sample_image_base64)} characters")
    
    try:
        print("\n🔄 Sending POST request...")
        start_time = time.time()
        
        response = requests.post(
            url, 
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"⏱️  Request completed in {processing_time:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Detection completed successfully")
            print("\n📋 Response Summary:")
            print(f"   Detection ID: {result.get('detection_id', 'N/A')}")
            print(f"   Status: {result.get('status', 'N/A')}")
            print(f"   Message: {result.get('message', 'N/A')}")
            
            # Show detection details if available
            if 'detection_result' in result:
                detection = result['detection_result']
                print(f"\n🔍 Detection Details:")
                print(f"   Billboard detected: {detection.get('billboard_detected', 'N/A')}")
                print(f"   Content extracted: {detection.get('content_extracted', 'N/A')}")
                print(f"   Safety analysis: {detection.get('safety_analysis', 'N/A')}")
                print(f"   Location compliance: {detection.get('location_compliance', 'N/A')}")
                print(f"   Size compliance: {detection.get('size_compliance', 'N/A')}")
            
            # Show full response
            print(f"\n📄 Full Response:")
            print(json.dumps(result, indent=2))
            
        else:
            print(f"❌ ERROR! Request failed with status {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Error text: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR! Is the server running on http://localhost:8000?")
        print("💡 Try starting the server with: python main.py")
    except requests.exceptions.Timeout:
        print("⏰ TIMEOUT ERROR! Request took too long (>30 seconds)")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")

def test_multiple_requests():
    """Test multiple detection requests with different coordinates"""
    
    locations = [
        {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
        {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
        {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946},
    ]
    
    print("🚀 Testing multiple detection requests...")
    
    for i, location in enumerate(locations, 1):
        print(f"\n--- Test {i}: {location['name']} ---")
        
        sample_image_base64 = create_sample_image()
        
        test_data = {
            "user_id": i,                     # Different user IDs
            "latitude": location["lat"],
            "longitude": location["lon"],
            "dimension": f"{15+i*2}x{8+i}",   # Different dimensions
            "quality_score": 0.7 + (i * 0.1), # Different quality scores
            "image_bytes": sample_image_base64
        }
        
        try:
            response = requests.post(
                "http://localhost:8000/api/billboards/detections/",
                json=test_data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {location['name']}: SUCCESS - Detection ID: {result.get('detection_id')}")
            else:
                print(f"❌ {location['name']}: FAILED - Status: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {location['name']}: ERROR - {e}")
        
        # Small delay between requests
        time.sleep(1)

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 BILLBOARD DETECTION API TESTER")
    print("=" * 60)
    
    # Test single request
    test_detection_endpoint()
    
    # Ask if user wants to test multiple requests
    print("\n" + "=" * 60)
    try:
        choice = input("🤔 Test multiple requests? (y/n): ").lower()
        if choice == 'y':
            test_multiple_requests()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    
    print("\n✨ Testing completed!")
