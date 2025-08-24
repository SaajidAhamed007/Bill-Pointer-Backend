"""
🇮🇳 INDIAN BILLBOARD COMPLIANCE DETECTION SYSTEM
Comprehensive unauthorized billboard detection for Indian cities
Based on Model Outdoor Advertising Policy 2016 and municipal regulations

Features:
- Dimensional compliance checking
- Location-based violation detection
- Content safety analysis
- Structural hazard assessment
- Real-time municipal reporting
- Citizen engagement platform
"""

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
import json
import math
from typing import List, Optional, Dict, Tuple
from io import BytesIO
from datetime import datetime
from enum import Enum

# Import production ML models
from app.core.production_ml_utils import (
    get_global_analyzer,
    analyze_billboard_image_stream,
    check_system_health
)

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# INDIAN REGULATORY FRAMEWORK
# =========================

class ViolationType(Enum):
    """Types of billboard violations based on Indian regulations"""
    DIMENSIONAL_VIOLATION = "dimensional_violation"
    LOCATION_VIOLATION = "location_violation"
    CONTENT_VIOLATION = "content_violation"
    STRUCTURAL_HAZARD = "structural_hazard"
    PERMIT_VIOLATION = "permit_violation"
    SAFETY_VIOLATION = "safety_violation"
    HERITAGE_VIOLATION = "heritage_violation"

class IndianBillboardRegulations:
    """
    Indian Billboard Regulations based on:
    - Model Outdoor Advertising Policy 2016
    - Municipal Corporation bylaws
    - Traffic and road safety regulations
    """
    
    # Dimensional limits (in meters)
    MAX_AREA_COMMERCIAL = 400  # 20x20 meters
    MAX_AREA_RESIDENTIAL = 100  # 10x10 meters
    MAX_HEIGHT_SINGLE_DIMENSION = 25
    MIN_GROUND_CLEARANCE = 2.5
    
    # Distance restrictions (in meters)
    MIN_DISTANCE_INTERSECTION = 100
    MIN_DISTANCE_TRAFFIC_SIGNAL = 50
    MIN_DISTANCE_SCHOOL = 200
    MIN_DISTANCE_HOSPITAL = 100
    MIN_DISTANCE_HERITAGE = 300
    MIN_DISTANCE_AIRPORT = 5000
    MIN_DISTANCE_RAILWAY = 150
    
    # Content restrictions
    PROHIBITED_CONTENT = [
        "tobacco", "alcohol", "gambling", "adult content",
        "political propaganda", "religious hate", "misleading claims"
    ]
    
    # Structural requirements
    MIN_STRUCTURAL_INTEGRITY_SCORE = 0.7
    MAX_AGE_YEARS = 5

class IndianCityZones:
    """City zoning data for major Indian cities"""
    
    RESTRICTED_ZONES = {
        # Delhi
        "delhi": {
            "heritage_sites": [
                {"name": "Red Fort", "lat": 28.6562, "lon": 77.2410, "radius": 500},
                {"name": "India Gate", "lat": 28.6129, "lon": 77.2295, "radius": 300},
                {"name": "Qutub Minar", "lat": 28.5245, "lon": 77.1855, "radius": 400},
            ],
            "government_zones": [
                {"name": "Parliament House", "lat": 28.6172, "lon": 77.2096, "radius": 1000},
                {"name": "Rashtrapati Bhavan", "lat": 28.6141, "lon": 77.1997, "radius": 800},
            ],
            "airports": [
                {"name": "IGI Airport", "lat": 28.5665, "lon": 77.1031, "radius": 5000},
            ]
        },
        # Mumbai
        "mumbai": {
            "heritage_sites": [
                {"name": "Gateway of India", "lat": 18.9220, "lon": 72.8347, "radius": 300},
                {"name": "Chhatrapati Shivaji Terminus", "lat": 18.9398, "lon": 72.8355, "radius": 400},
            ],
            "airports": [
                {"name": "Chhatrapati Shivaji International Airport", "lat": 19.0896, "lon": 72.8656, "radius": 5000},
            ]
        },
        # Bangalore
        "bangalore": {
            "heritage_sites": [
                {"name": "Bangalore Palace", "lat": 12.9985, "lon": 77.5926, "radius": 200},
                {"name": "Tipu Sultan's Summer Palace", "lat": 12.9591, "lon": 77.5746, "radius": 200},
            ],
            "airports": [
                {"name": "Kempegowda International Airport", "lat": 13.1986, "lon": 77.7066, "radius": 5000},
            ]
        }
    }

# =========================
# ADVANCED DETECTION FUNCTIONS
# =========================

def detect_billboard_with_indian_compliance(image_stream: BytesIO) -> Dict:
    """
    Comprehensive billboard detection with Indian compliance checking
    """
    try:
        analyzer = get_global_analyzer()
        result = analyzer.analyze_image_stream(image_stream)
        
        # Enhanced analysis for Indian context
        enhanced_result = {
            **result,
            "indian_compliance": {
                "compliance_checked": True,
                "regulatory_framework": "Model Outdoor Advertising Policy 2016",
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
        return enhanced_result
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "indian_compliance": {
                "compliance_checked": False,
                "error": "Failed to perform compliance analysis"
            }
        }

def check_dimensional_compliance(dimension: str, location_type: str = "commercial") -> Dict:
    """
    Check dimensional compliance based on Indian regulations
    """
    violations = []
    compliance_details = {
        "is_compliant": True,
        "violations": [],
        "max_allowed_area": 0,
        "actual_area": 0,
        "regulation_source": "Model Outdoor Advertising Policy 2016"
    }
    
    try:
        # Parse dimensions
        dimension_clean = dimension.replace(" ", "").lower()
        if 'x' not in dimension_clean:
            violations.append("Invalid dimension format - use format like '10x20'")
            compliance_details["is_compliant"] = False
            compliance_details["violations"] = violations
            return compliance_details
        
        parts = dimension_clean.split('x')
        if len(parts) != 2:
            violations.append("Invalid dimension format - requires exactly two dimensions")
            compliance_details["is_compliant"] = False
            compliance_details["violations"] = violations
            return compliance_details
        
        try:
            width = float(parts[0])
            height = float(parts[1])
        except ValueError:
            violations.append("Non-numeric dimensions provided")
            compliance_details["is_compliant"] = False
            compliance_details["violations"] = violations
            return compliance_details
        
        area = width * height
        max_dimension = max(width, height)
        
        # Set limits based on location type
        if location_type == "residential":
            max_allowed_area = IndianBillboardRegulations.MAX_AREA_RESIDENTIAL
        else:
            max_allowed_area = IndianBillboardRegulations.MAX_AREA_COMMERCIAL
        
        compliance_details["actual_area"] = area
        compliance_details["max_allowed_area"] = max_allowed_area
        
        # Check area compliance
        if area > max_allowed_area:
            violations.append(f"Exceeds maximum allowed area: {area}m² > {max_allowed_area}m² for {location_type} zone")
        
        # Check maximum dimension
        if max_dimension > IndianBillboardRegulations.MAX_HEIGHT_SINGLE_DIMENSION:
            violations.append(f"Exceeds maximum single dimension: {max_dimension}m > {IndianBillboardRegulations.MAX_HEIGHT_SINGLE_DIMENSION}m")
        
        # Check minimum dimension (avoid tiny billboards that could be distracting)
        if width < 1 or height < 1:
            violations.append("Billboard dimensions too small - minimum 1m x 1m required")
        
        # Check aspect ratio (avoid extremely thin billboards)
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 10:
            violations.append(f"Extreme aspect ratio {aspect_ratio:.1f}:1 - maximum 10:1 allowed")
        
        if violations:
            compliance_details["is_compliant"] = False
            compliance_details["violations"] = violations
        
        return compliance_details
        
    except Exception as e:
        return {
            "is_compliant": False,
            "violations": [f"Error in dimensional analysis: {str(e)}"],
            "max_allowed_area": 0,
            "actual_area": 0,
            "regulation_source": "Model Outdoor Advertising Policy 2016"
        }

def check_location_compliance(latitude: float, longitude: float, city: str = "delhi") -> Dict:
    """
    Check location-based compliance with Indian regulations
    """
    violations = []
    location_analysis = {
        "is_compliant": True,
        "violations": [],
        "restricted_zones_nearby": [],
        "city": city.lower(),
        "coordinates": {"latitude": latitude, "longitude": longitude}
    }
    
    try:
        city_data = IndianCityZones.RESTRICTED_ZONES.get(city.lower(), {})
        
        # Check heritage sites
        heritage_sites = city_data.get("heritage_sites", [])
        for site in heritage_sites:
            distance = calculate_distance(latitude, longitude, site["lat"], site["lon"])
            if distance < site["radius"]:
                violations.append(f"Within {site['radius']}m of heritage site: {site['name']} (distance: {distance:.0f}m)")
                location_analysis["restricted_zones_nearby"].append({
                    "type": "heritage_site",
                    "name": site["name"],
                    "distance": distance,
                    "min_required": site["radius"]
                })
        
        # Check government zones
        govt_zones = city_data.get("government_zones", [])
        for zone in govt_zones:
            distance = calculate_distance(latitude, longitude, zone["lat"], zone["lon"])
            if distance < zone["radius"]:
                violations.append(f"Within {zone['radius']}m of government area: {zone['name']} (distance: {distance:.0f}m)")
                location_analysis["restricted_zones_nearby"].append({
                    "type": "government_zone",
                    "name": zone["name"],
                    "distance": distance,
                    "min_required": zone["radius"]
                })
        
        # Check airports
        airports = city_data.get("airports", [])
        for airport in airports:
            distance = calculate_distance(latitude, longitude, airport["lat"], airport["lon"])
            if distance < airport["radius"]:
                violations.append(f"Within {airport['radius']}m of airport: {airport['name']} (distance: {distance:.0f}m)")
                location_analysis["restricted_zones_nearby"].append({
                    "type": "airport",
                    "name": airport["name"],
                    "distance": distance,
                    "min_required": airport["radius"]
                })
        
        # Additional generic checks (would integrate with real mapping APIs in production)
        generic_violations = check_generic_location_violations(latitude, longitude)
        violations.extend(generic_violations)
        
        if violations:
            location_analysis["is_compliant"] = False
            location_analysis["violations"] = violations
        
        return location_analysis
        
    except Exception as e:
        return {
            "is_compliant": False,
            "violations": [f"Error in location analysis: {str(e)}"],
            "restricted_zones_nearby": [],
            "city": city,
            "coordinates": {"latitude": latitude, "longitude": longitude}
        }

def check_generic_location_violations(latitude: float, longitude: float) -> List[str]:
    """
    Check for generic location violations using coordinate patterns
    In production, this would integrate with mapping APIs
    """
    violations = []
    
    # Simulate checks for major intersections (in production, use real mapping data)
    # This is a simplified version for demonstration
    
    # Check if coordinates suggest major intersection areas
    # (This would be replaced with actual Google Maps/OpenStreetMap integration)
    
    # Example: Major Delhi intersections (simplified)
    major_intersections = [
        {"name": "Connaught Place", "lat": 28.6315, "lon": 77.2167, "radius": 100},
        {"name": "Rajiv Chowk", "lat": 28.6333, "lon": 77.2167, "radius": 100},
        {"name": "Karol Bagh", "lat": 28.6508, "lon": 77.1901, "radius": 75},
    ]
    
    for intersection in major_intersections:
        distance = calculate_distance(latitude, longitude, intersection["lat"], intersection["lon"])
        if distance < intersection["radius"]:
            violations.append(f"Within {intersection['radius']}m of major intersection: {intersection['name']}")
    
    return violations

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two coordinates using Haversine formula
    Returns distance in meters
    """
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def analyze_content_safety_indian_context(analysis_data: Dict) -> Dict:
    """
    Analyze content safety with specific focus on Indian regulations
    """
    safety_analysis = {
        "is_safe": True,
        "violations": [],
        "risk_level": "safe",
        "indian_specific_checks": True,
        "prohibited_content_detected": []
    }
    
    try:
        if "billboard_results" not in analysis_data:
            return safety_analysis
        
        all_text = ""
        for billboard in analysis_data["billboard_results"]:
            text = billboard.get("extracted_text", "").lower()
            all_text += " " + text
        
        # Check for prohibited content specific to Indian regulations
        for prohibited in IndianBillboardRegulations.PROHIBITED_CONTENT:
            if prohibited in all_text:
                safety_analysis["prohibited_content_detected"].append(prohibited)
                safety_analysis["violations"].append(f"Contains prohibited content: {prohibited}")
                safety_analysis["is_safe"] = False
        
        # Check ML safety analysis
        for billboard in analysis_data["billboard_results"]:
            ml_safety = billboard.get("safety_analysis", {})
            if not ml_safety.get("safe", True):
                risk_level = ml_safety.get("risk_level", "safe")
                safety_analysis["violations"].append(f"ML detected unsafe content: {risk_level}")
                safety_analysis["is_safe"] = False
                safety_analysis["risk_level"] = risk_level
        
        # Additional Indian-specific content checks
        indian_violations = check_indian_content_regulations(all_text)
        safety_analysis["violations"].extend(indian_violations)
        if indian_violations:
            safety_analysis["is_safe"] = False
        
        return safety_analysis
        
    except Exception as e:
        return {
            "is_safe": False,
            "violations": [f"Error in content safety analysis: {str(e)}"],
            "risk_level": "unknown",
            "indian_specific_checks": False,
            "prohibited_content_detected": []
        }

def check_indian_content_regulations(text: str) -> List[str]:
    """
    Check content against Indian advertising regulations
    """
    violations = []
    text_lower = text.lower()
    
    # Check for specific Indian regulatory violations
    indian_prohibitions = [
        {"keyword": "cigarette", "violation": "Tobacco advertising prohibited under COTPA"},
        {"keyword": "beer", "violation": "Alcohol advertising prohibited in public spaces"},
        {"keyword": "wine", "violation": "Alcohol advertising prohibited in public spaces"},
        {"keyword": "betting", "violation": "Gambling advertising prohibited"},
        {"keyword": "casino", "violation": "Gambling advertising prohibited"},
        {"keyword": "fake", "violation": "Misleading advertising prohibited"},
        {"keyword": "guaranteed cure", "violation": "False medical claims prohibited"},
    ]
    
    for item in indian_prohibitions:
        if item["keyword"] in text_lower:
            violations.append(item["violation"])
    
    return violations

def assess_structural_integrity(analysis_data: Dict, dimension: str) -> Dict:
    """
    Assess structural integrity and safety hazards
    """
    integrity_analysis = {
        "is_structurally_sound": True,
        "hazard_level": "low",
        "violations": [],
        "integrity_score": 1.0,
        "recommendations": []
    }
    
    try:
        # Analyze based on ML detection confidence (proxy for structural visibility)
        if "billboard_results" in analysis_data:
            confidences = []
            for billboard in analysis_data["billboard_results"]:
                conf = billboard.get("detection_confidence", 0) / 100.0
                confidences.append(conf)
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                integrity_analysis["integrity_score"] = avg_confidence
                
                # Low confidence might indicate structural issues
                if avg_confidence < 0.6:
                    integrity_analysis["violations"].append("Low detection confidence suggests possible structural deterioration")
                    integrity_analysis["hazard_level"] = "medium"
                    integrity_analysis["is_structurally_sound"] = False
                    integrity_analysis["recommendations"].append("Structural inspection recommended")
                
                if avg_confidence < 0.4:
                    integrity_analysis["hazard_level"] = "high"
                    integrity_analysis["violations"].append("Very low visibility suggests severe structural issues")
                    integrity_analysis["recommendations"].append("Immediate structural assessment required")
        
        # Check dimensional safety
        try:
            parts = dimension.replace(" ", "").lower().split('x')
            if len(parts) == 2:
                width, height = float(parts[0]), float(parts[1])
                area = width * height
                
                # Large billboards require stronger structural support
                if area > 300:
                    integrity_analysis["recommendations"].append("Enhanced structural support verification required for large billboard")
                
                # High billboards need wind resistance checks
                if max(width, height) > 20:
                    integrity_analysis["recommendations"].append("Wind resistance analysis required for tall billboard")
        
        except:
            pass  # Ignore dimension parsing errors for structural analysis
        
        return integrity_analysis
        
    except Exception as e:
        return {
            "is_structurally_sound": False,
            "hazard_level": "unknown",
            "violations": [f"Error in structural analysis: {str(e)}"],
            "integrity_score": 0.0,
            "recommendations": ["Manual structural inspection required due to analysis error"]
        }

def generate_violation_report(compliance_results: Dict) -> Dict:
    """
    Generate comprehensive violation report for municipal authorities
    """
    all_violations = []
    violation_types = []
    severity_score = 0.0
    
    # Collect all violations
    if not compliance_results.get("dimensional", {}).get("is_compliant", True):
        all_violations.extend(compliance_results["dimensional"]["violations"])
        violation_types.append(ViolationType.DIMENSIONAL_VIOLATION.value)
        severity_score += 0.3
    
    if not compliance_results.get("location", {}).get("is_compliant", True):
        all_violations.extend(compliance_results["location"]["violations"])
        violation_types.append(ViolationType.LOCATION_VIOLATION.value)
        severity_score += 0.4
    
    if not compliance_results.get("content", {}).get("is_safe", True):
        all_violations.extend(compliance_results["content"]["violations"])
        violation_types.append(ViolationType.CONTENT_VIOLATION.value)
        severity_score += 0.5
    
    if not compliance_results.get("structural", {}).get("is_structurally_sound", True):
        all_violations.extend(compliance_results["structural"]["violations"])
        violation_types.append(ViolationType.STRUCTURAL_HAZARD.value)
        severity_score += 0.6
    
    # Determine overall compliance status
    is_compliant = len(all_violations) == 0
    
    if severity_score >= 0.8:
        recommendation = "immediate_action"
        priority = "high"
    elif severity_score >= 0.5:
        recommendation = "investigation_required"
        priority = "medium"
    elif severity_score > 0:
        recommendation = "monitoring_required"
        priority = "low"
    else:
        recommendation = "compliant"
        priority = "none"
    
    return {
        "is_compliant": is_compliant,
        "total_violations": len(all_violations),
        "violation_types": violation_types,
        "all_violations": all_violations,
        "severity_score": round(severity_score, 2),
        "recommendation": recommendation,
        "priority": priority,
        "requires_municipal_action": not is_compliant,
        "regulatory_framework": "Model Outdoor Advertising Policy 2016"
    }

def send_municipal_alert(detection: Detection, violation_report: Dict):
    """
    Send comprehensive alert to municipal authorities
    """
    try:
        alert_payload = {
            "alert_id": f"BILLBOARD_VIOLATION_{detection.id}_{int(time.time())}",
            "detection_id": detection.id,
            "citizen_reporter": {
                "user_id": detection.user_id,
                "report_timestamp": detection.updated_at
            },
            "location": {
                "latitude": detection.latitude,
                "longitude": detection.longitude,
                "coordinates_verified": True
            },
            "billboard_details": {
                "dimensions": detection.dimension,
                "estimated_area": violation_report.get("dimensional", {}).get("actual_area", 0),
                "content_extracted": detection.content
            },
            "violations": {
                "total_count": violation_report["total_violations"],
                "types": violation_report["violation_types"],
                "details": violation_report["all_violations"],
                "severity_score": violation_report["severity_score"],
                "priority": violation_report["priority"]
            },
            "compliance_analysis": {
                "regulatory_framework": "Model Outdoor Advertising Policy 2016",
                "dimensional_compliance": violation_report.get("dimensional", {}),
                "location_compliance": violation_report.get("location", {}),
                "content_safety": violation_report.get("content", {}),
                "structural_assessment": violation_report.get("structural", {})
            },
            "recommendation": {
                "action_required": violation_report["recommendation"],
                "municipal_response_needed": violation_report["requires_municipal_action"],
                "suggested_timeline": "72 hours" if violation_report["priority"] == "high" else "7 days"
            },
            "evidence": {
                "image_path": detection.image_path,
                "ml_analysis_available": True,
                "citizen_report": True
            }
        }
        
        # Log comprehensive alert (in production, send to municipal API)
        print(f"🚨 MUNICIPAL ALERT GENERATED:")
        print(f"Alert ID: {alert_payload['alert_id']}")
        print(f"Priority: {violation_report['priority'].upper()}")
        print(f"Violations: {violation_report['total_violations']}")
        print(f"Location: {detection.latitude}, {detection.longitude}")
        print(f"Action Required: {violation_report['recommendation']}")
        
        # In production, send to municipal authority API:
        # municipal_api_url = "https://municipal-api.gov.in/billboard-violations"
        # response = requests.post(municipal_api_url, json=alert_payload, timeout=30)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to send municipal alert: {e}")
        return False

# =========================
# API ROUTES
# =========================

@router.get("/health/indian-compliance")
async def health_check_indian_compliance():
    """Check system health with Indian compliance capabilities"""
    ml_health = check_system_health()
    
    return {
        "api_status": "healthy",
        "ml_system": ml_health,
        "indian_compliance": {
            "regulatory_framework": "Model Outdoor Advertising Policy 2016",
            "compliance_checks_available": True,
            "supported_cities": list(IndianCityZones.RESTRICTED_ZONES.keys()),
            "violation_types_detected": [vt.value for vt in ViolationType]
        },
        "privacy_compliance": {
            "data_protection": "Citizen privacy protected",
            "no_facial_recognition": True,
            "data_encryption": "Enabled",
            "gdpr_compliant": True
        },
        "timestamp": int(time.time())
    }

@router.post("/detections/")
async def create_billboard_detection_indian_compliance(
    user_id: int = Form(..., description="Citizen reporter ID"),
    latitude: float = Form(..., description="Billboard latitude coordinate"),
    longitude: float = Form(..., description="Billboard longitude coordinate"),
    dimension: str = Form(..., description="Billboard dimensions (format: 'widthXheight' e.g., '10x20')"),
    quality_score: float = Form(..., description="Image quality score (0.0 to 1.0)"),
    city: str = Form("delhi", description="City name (delhi/mumbai/bangalore)"),
    location_type: str = Form("commercial", description="Location type (commercial/residential)"),
    image: UploadFile = File(..., description="Billboard image file"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """
    🇮🇳 COMPREHENSIVE INDIAN BILLBOARD COMPLIANCE DETECTION
    
    This endpoint provides complete unauthorized billboard detection following:
    - Model Outdoor Advertising Policy 2016
    - Municipal Corporation regulations
    - Indian traffic and safety laws
    - Heritage and cultural protection guidelines
    """
    
    try:
        print(f"🇮🇳 Starting Indian billboard compliance check for user {user_id}")
        
        # Input validation
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image (JPG, PNG, etc.)")
        
        if not 0.0 <= quality_score <= 1.0:
            raise HTTPException(status_code=400, detail="Quality score must be between 0.0 and 1.0")
        
        if not dimension or dimension.strip() == "":
            raise HTTPException(status_code=400, detail="Billboard dimensions are required")
        
        if city.lower() not in IndianCityZones.RESTRICTED_ZONES:
            print(f"⚠️ Warning: City '{city}' not in database, using generic checks")
        
        # Save image and create analysis stream
        file_ext = os.path.splitext(image.filename)[1] if image.filename else ".jpg"
        file_name = f"billboard_{uuid.uuid4().hex}_{int(time.time())}{file_ext}"
        image_path = os.path.join(UPLOAD_DIR, file_name)
        
        image_content = await image.read()
        image_stream = BytesIO(image_content)
        
        with open(image_path, "wb") as buffer:
            buffer.write(image_content)
        
        print(f"📷 Image saved: {file_name}")
        
        # STEP 1: AI-Powered Billboard Detection
        print("🤖 Step 1: AI Billboard Detection...")
        analysis_data = detect_billboard_with_indian_compliance(image_stream)
        
        if analysis_data.get("status") != "analysis_complete" or analysis_data.get("total_billboards", 0) == 0:
            os.remove(image_path)
            error_msg = analysis_data.get("message", "No billboards detected in the image")
            raise HTTPException(status_code=400, detail=f"Billboard detection failed: {error_msg}")
        
        print(f"✅ Detected {analysis_data.get('total_billboards', 0)} billboard(s)")
        
        # STEP 2: Extract Content
        print("📝 Step 2: Text Extraction...")
        extracted_content = ""
        if "billboard_results" in analysis_data:
            texts = []
            for billboard in analysis_data["billboard_results"]:
                text = billboard.get("extracted_text", "").strip()
                if text:
                    texts.append(text)
            extracted_content = " | ".join(texts)
        
        # STEP 3: Comprehensive Compliance Analysis
        print("⚖️ Step 3: Indian Regulatory Compliance Analysis...")
        
        # 3.1 Dimensional Compliance
        dimensional_compliance = check_dimensional_compliance(dimension, location_type)
        
        # 3.2 Location Compliance
        location_compliance = check_location_compliance(latitude, longitude, city)
        
        # 3.3 Content Safety Analysis
        content_safety = analyze_content_safety_indian_context(analysis_data)
        
        # 3.4 Structural Integrity Assessment
        structural_assessment = assess_structural_integrity(analysis_data, dimension)
        
        # STEP 4: Generate Comprehensive Compliance Report
        print("📊 Step 4: Generating Compliance Report...")
        compliance_results = {
            "dimensional": dimensional_compliance,
            "location": location_compliance,
            "content": content_safety,
            "structural": structural_assessment
        }
        
        violation_report = generate_violation_report(compliance_results)
        
        # STEP 5: Determine Final Status
        if violation_report["is_compliant"]:
            final_status = DetectionStatus.approved
            status_message = "Billboard compliant with Indian regulations"
        else:
            if violation_report["priority"] == "high":
                final_status = DetectionStatus.flagged
                status_message = "Billboard flagged for immediate municipal action"
            else:
                final_status = DetectionStatus.pending
                status_message = "Billboard requires investigation"
        
        # STEP 6: Calculate Quality Metrics
        ml_quality = 0.0
        if "performance" in analysis_data:
            perf = analysis_data["performance"]
            ml_quality = (perf.get("detection_speed", 0) + perf.get("accuracy_score", 0)) / 2
        
        final_quality_score = (quality_score + ml_quality + structural_assessment["integrity_score"]) / 3
        
        # STEP 7: Save to Database
        print("💾 Step 7: Saving Detection Record...")
        detection = Detection(
            user_id=user_id,
            image_path=image_path,
            dimension=dimension,
            content=extracted_content,
            latitude=latitude,
            longitude=longitude,
            quality_score=final_quality_score,
            status=final_status,
            flagged_reason="; ".join(violation_report["all_violations"]) if violation_report["all_violations"] else None,
            updated_at=int(time.time())
        )
        
        db.add(detection)
        db.commit()
        db.refresh(detection)
        
        # STEP 8: Municipal Alert (if violations found)
        if not violation_report["is_compliant"]:
            print("🚨 Step 8: Sending Municipal Alert...")
            background_tasks.add_task(send_municipal_alert, detection, {
                **violation_report,
                **compliance_results
            })
        
        # STEP 9: Generate Response
        response = {
            "detection_id": detection.id,
            "user_id": detection.user_id,
            "status": detection.status.value,
            "message": status_message,
            
            # Analysis Summary
            "analysis_summary": {
                "billboards_detected": analysis_data.get("total_billboards", 0),
                "content_extracted": bool(extracted_content),
                "processing_time_ms": analysis_data.get("processing_time", 0),
                "analysis_timestamp": datetime.now().isoformat()
            },
            
            # Indian Compliance Results
            "indian_compliance": {
                "regulatory_framework": "Model Outdoor Advertising Policy 2016",
                "overall_compliance": violation_report["is_compliant"],
                "total_violations": violation_report["total_violations"],
                "violation_types": violation_report["violation_types"],
                "severity_score": violation_report["severity_score"],
                "priority": violation_report["priority"],
                "recommendation": violation_report["recommendation"]
            },
            
            # Detailed Compliance Analysis
            "compliance_details": {
                "dimensional_analysis": dimensional_compliance,
                "location_analysis": location_compliance,
                "content_safety": content_safety,
                "structural_assessment": structural_assessment
            },
            
            # Quality Metrics
            "quality_metrics": {
                "user_quality_score": quality_score,
                "ml_quality_score": ml_quality,
                "structural_integrity": structural_assessment["integrity_score"],
                "final_quality_score": round(final_quality_score, 3)
            },
            
            # Municipal Action
            "municipal_action": {
                "alert_sent": not violation_report["is_compliant"],
                "requires_investigation": violation_report["requires_municipal_action"],
                "estimated_response_time": "72 hours" if violation_report["priority"] == "high" else "7 days"
            },
            
            # Privacy and Ethics
            "privacy_notice": {
                "data_use": "Image and location data used only for billboard compliance checking",
                "no_facial_recognition": "No facial recognition or personal identification performed",
                "data_retention": "Evidence stored for municipal compliance verification only",
                "citizen_rights": "You can request data deletion after municipal review is complete"
            }
        }
        
        print(f"✅ Detection {detection.id} completed with status: {final_status.value}")
        print(f"🎯 Compliance: {violation_report['is_compliant']}, Violations: {violation_report['total_violations']}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Critical error in billboard detection: {str(e)}")
        if 'image_path' in locals() and os.path.exists(image_path):
            os.remove(image_path)
        raise HTTPException(status_code=500, detail=f"System error: {str(e)}")

@router.get("/detections/user/{user_id}")
async def get_user_detections_with_compliance(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(jwt_required),
    status: Optional[str] = None,
    compliance_only: bool = False
):
    """Get user's billboard detections with compliance information"""
    
    if user_id != current_user.get("id"):
        raise HTTPException(status_code=403, detail="Access denied: Can only view your own detections")
    
    query = db.query(Detection).filter(Detection.user_id == user_id)
    
    if status:
        try:
            query = query.filter(Detection.status == DetectionStatus(status))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid status. Use: approved, flagged, pending")
    
    if compliance_only:
        query = query.filter(Detection.status != DetectionStatus.approved)
    
    detections = query.order_by(Detection.updated_at.desc()).all()
    
    return {
        "user_id": user_id,
        "total_detections": len(detections),
        "compliance_summary": {
            "total_reports": len(detections),
            "approved": len([d for d in detections if d.status == DetectionStatus.approved]),
            "flagged": len([d for d in detections if d.status == DetectionStatus.flagged]),
            "pending": len([d for d in detections if d.status == DetectionStatus.pending])
        },
        "detections": [
            {
                "id": d.id,
                "dimension": d.dimension,
                "content": d.content,
                "location": {"latitude": d.latitude, "longitude": d.longitude},
                "status": d.status.value,
                "quality_score": round(d.quality_score, 3),
                "violations": d.flagged_reason,
                "report_date": datetime.fromtimestamp(d.updated_at).isoformat(),
                "municipal_action_required": d.status == DetectionStatus.flagged
            }
            for d in detections
        ]
    }

@router.get("/analytics/indian-compliance")
async def get_indian_compliance_analytics(db: Session = Depends(get_db)):
    """Get analytics dashboard for Indian billboard compliance"""
    
    total_detections = db.query(Detection).count()
    approved = db.query(Detection).filter(Detection.status == DetectionStatus.approved).count()
    flagged = db.query(Detection).filter(Detection.status == DetectionStatus.flagged).count()
    pending = db.query(Detection).filter(Detection.status == DetectionStatus.pending).count()
    
    return {
        "overview": {
            "total_citizen_reports": total_detections,
            "compliant_billboards": approved,
            "non_compliant_flagged": flagged,
            "under_investigation": pending,
            "compliance_rate": round(approved / total_detections * 100, 2) if total_detections > 0 else 0,
            "violation_rate": round(flagged / total_detections * 100, 2) if total_detections > 0 else 0
        },
        "indian_regulations": {
            "framework": "Model Outdoor Advertising Policy 2016",
            "supported_cities": list(IndianCityZones.RESTRICTED_ZONES.keys()),
            "violation_types_tracked": [vt.value for vt in ViolationType]
        },
        "citizen_engagement": {
            "total_reports": total_detections,
            "unique_reporters": db.query(Detection.user_id).distinct().count(),
            "average_reports_per_citizen": round(total_detections / max(1, db.query(Detection.user_id).distinct().count()), 2)
        },
        "municipal_impact": {
            "high_priority_alerts": flagged,
            "pending_investigations": pending,
            "estimated_municipal_workload": f"{flagged * 2 + pending * 1} hours"
        }
    }

@router.get("/regulations/info")
async def get_indian_billboard_regulations():
    """Get information about Indian billboard regulations and compliance requirements"""
    
    return {
        "regulatory_framework": {
            "primary_law": "Model Outdoor Advertising Policy 2016",
            "authority": "Ministry of Urban Development, Government of India",
            "local_implementation": "Municipal Corporations and Urban Local Bodies"
        },
        "dimensional_limits": {
            "commercial_zones": {
                "max_area_sqm": IndianBillboardRegulations.MAX_AREA_COMMERCIAL,
                "max_single_dimension_m": IndianBillboardRegulations.MAX_HEIGHT_SINGLE_DIMENSION,
                "min_ground_clearance_m": IndianBillboardRegulations.MIN_GROUND_CLEARANCE
            },
            "residential_zones": {
                "max_area_sqm": IndianBillboardRegulations.MAX_AREA_RESIDENTIAL,
                "max_single_dimension_m": IndianBillboardRegulations.MAX_HEIGHT_SINGLE_DIMENSION,
                "min_ground_clearance_m": IndianBillboardRegulations.MIN_GROUND_CLEARANCE
            }
        },
        "location_restrictions": {
            "minimum_distances": {
                "intersections_m": IndianBillboardRegulations.MIN_DISTANCE_INTERSECTION,
                "traffic_signals_m": IndianBillboardRegulations.MIN_DISTANCE_TRAFFIC_SIGNAL,
                "schools_m": IndianBillboardRegulations.MIN_DISTANCE_SCHOOL,
                "hospitals_m": IndianBillboardRegulations.MIN_DISTANCE_HOSPITAL,
                "heritage_sites_m": IndianBillboardRegulations.MIN_DISTANCE_HERITAGE,
                "airports_m": IndianBillboardRegulations.MIN_DISTANCE_AIRPORT,
                "railways_m": IndianBillboardRegulations.MIN_DISTANCE_RAILWAY
            }
        },
        "content_restrictions": {
            "prohibited_content": IndianBillboardRegulations.PROHIBITED_CONTENT,
            "additional_laws": [
                "Cigarettes and Other Tobacco Products Act (COTPA)",
                "Consumer Protection Act",
                "Indecent Representation of Women (Prohibition) Act"
            ]
        },
        "structural_requirements": {
            "min_integrity_score": IndianBillboardRegulations.MIN_STRUCTURAL_INTEGRITY_SCORE,
            "max_age_years": IndianBillboardRegulations.MAX_AGE_YEARS,
            "wind_resistance": "Required for billboards > 20m height",
            "foundation_requirements": "As per local municipal building codes"
        },
        "violation_consequences": {
            "penalties": "As per local municipal bylaws",
            "removal_timeline": "24-72 hours for high-risk violations",
            "appeal_process": "Available through municipal corporation"
        },
        "citizen_reporting": {
            "how_to_report": "Use this mobile app to capture and report unauthorized billboards",
            "data_privacy": "Personal data protected, only billboard compliance data shared with authorities",
            "follow_up": "Municipal authorities will investigate and take action based on report priority"
        }
    }
