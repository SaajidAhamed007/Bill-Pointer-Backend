# PowerShell script to convert image to base64 and send POST request

# Function to convert image to base64
function Convert-ImageToBase64 {
    param (
        [string]$ImagePath
    )
    
    if (Test-Path $ImagePath) {
        $imageBytes = [System.IO.File]::ReadAllBytes($ImagePath)
        $base64String = [System.Convert]::ToBase64String($imageBytes)
        Write-Host "✅ Image converted to base64" -ForegroundColor Green
        Write-Host "📊 Original size: $($imageBytes.Length) bytes" -ForegroundColor Yellow
        Write-Host "📊 Base64 size: $($base64String.Length) characters" -ForegroundColor Yellow
        return $base64String
    } else {
        Write-Host "❌ Image file not found: $ImagePath" -ForegroundColor Red
        return $null
    }
}

# Function to send billboard detection request
function Send-BillboardDetectionRequest {
    param (
        [string]$ImageBase64,
        [int]$UserId = 1,
        [double]$Latitude = 28.6139,
        [double]$Longitude = 77.2090,
        [string]$Dimension = "10x5",
        [double]$QualityScore = 0.8
    )
    
    $url = "http://127.0.0.1:8000/api/billboards/detections/"
    
    $requestBody = @{
        user_id = $UserId
        latitude = $Latitude
        longitude = $Longitude
        dimension = $Dimension
        quality_score = $QualityScore
        image_bytes = $ImageBase64
    } | ConvertTo-Json
    
    $headers = @{
        "Content-Type" = "application/json"
    }
    
    try {
        Write-Host "🚀 Sending POST request to billboard detection API..." -ForegroundColor Cyan
        
        $response = Invoke-RestMethod -Uri $url -Method POST -Headers $headers -Body $requestBody
        
        Write-Host "✅ SUCCESS! Billboard detection completed" -ForegroundColor Green
        Write-Host "🆔 Detection ID: $($response.id)" -ForegroundColor Yellow
        Write-Host "📍 Status: $($response.status)" -ForegroundColor Yellow
        Write-Host "💬 Message: $($response.message)" -ForegroundColor Yellow
        
        if ($response.billboard_content) {
            Write-Host "📝 Extracted Text: $($response.billboard_content.extracted_text)" -ForegroundColor Yellow
            Write-Host "🛡️ Content Safe: $($response.billboard_content.content_appropriate)" -ForegroundColor Yellow
        }
        
        return $response
        
    } catch {
        Write-Host "❌ Error sending request: $($_.Exception.Message)" -ForegroundColor Red
        
        if ($_.Exception.Response) {
            $statusCode = $_.Exception.Response.StatusCode
            Write-Host "📊 Status Code: $statusCode" -ForegroundColor Red
            
            try {
                $errorContent = $_.Exception.Response.Content.ReadAsStringAsync().Result
                Write-Host "📄 Error Details: $errorContent" -ForegroundColor Red
            } catch {
                Write-Host "📄 Could not read error details" -ForegroundColor Red
            }
        }
        
        return $null
    }
}

# Main execution
Write-Host "🎯 Billboard Detection - PowerShell Image Upload Test" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# Example usage:

# Method 1: Use an existing image file
# $imagePath = "sample_billboard.jpg"  # Replace with your image path
# $imageBase64 = Convert-ImageToBase64 -ImagePath $imagePath

# Method 2: Create a simple test base64 string (minimal valid JPEG)
Write-Host "📝 Using minimal test image..." -ForegroundColor Yellow
$imageBase64 = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/gA=="

if ($imageBase64) {
    $result = Send-BillboardDetectionRequest -ImageBase64 $imageBase64
    
    if ($result) {
        Write-Host "`n🎉 Request completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "`n❌ Request failed" -ForegroundColor Red
    }
} else {
    Write-Host "❌ No image data available" -ForegroundColor Red
}

Write-Host "`n" + "=" * 60 -ForegroundColor Cyan
Write-Host "💡 To use your own image:" -ForegroundColor Cyan
Write-Host "1. Put your image file in this folder" -ForegroundColor White
Write-Host "2. Uncomment and update the imagePath variable" -ForegroundColor White
Write-Host "3. Run this script again" -ForegroundColor White
