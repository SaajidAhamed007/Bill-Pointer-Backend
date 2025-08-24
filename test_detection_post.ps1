# PowerShell script to test Billboard Detection API
# Simple POST request to detection endpoint

Write-Host "🚀 Testing Billboard Detection API with PowerShell" -ForegroundColor Green

# Create a simple base64 encoded image (1x1 pixel PNG)
$base64Image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

# Test data
$testData = @{
    user_id = 1
    latitude = 28.6139
    longitude = 77.2090
    dimension = "20x10"
    quality_score = 0.8
    image_bytes = $base64Image
} | ConvertTo-Json

Write-Host "📊 Request Data:" -ForegroundColor Yellow
Write-Host $testData

Write-Host "`n🔄 Sending POST request to http://localhost:8000/api/billboards/detections/..." -ForegroundColor Cyan

try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/api/billboards/detections/" -Method Post -Body $testData -ContentType "application/json" -TimeoutSec 30
    
    Write-Host "✅ SUCCESS! Response received:" -ForegroundColor Green
    Write-Host ($response | ConvertTo-Json -Depth 10) -ForegroundColor White
    
} catch {
    if ($_.Exception.Message -like "*Unable to connect*") {
        Write-Host "❌ CONNECTION ERROR! Is the server running?" -ForegroundColor Red
        Write-Host "💡 Start the server with: python main.py" -ForegroundColor Yellow
    } else {
        Write-Host "❌ ERROR: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "📄 Response: $($_.ErrorDetails.Message)" -ForegroundColor Yellow
    }
}

Write-Host "`n✨ Test completed!" -ForegroundColor Green
