#!/usr/bin/env bash
set -euo pipefail

# Test API endpoints
# Requires API server to be running

echo "=========================================="
echo "Test API Endpoints"
echo "=========================================="

API_URL="${API_URL:-http://localhost:8080}"

# Test health endpoint
echo ""
echo "Test 1: Health Check"
echo "===================="
curl -s "$API_URL/health" | python -m json.tool

# Test with sample image
echo ""
echo ""
echo "Test 2: Inspection Request"
echo "=========================="

# Check if test image exists
TEST_IMAGE="data/stcray/test/images/000000.jpg"
if [ ! -f "$TEST_IMAGE" ]; then
    echo "Warning: Test image not found: $TEST_IMAGE"
    echo "Using placeholder test"
    TEST_IMAGE=""
fi

if [ -n "$TEST_IMAGE" ]; then
    # Encode image to base64
    IMAGE_BASE64=$(base64 -i "$TEST_IMAGE" | tr -d '\n')
    
    # Send request
    curl -s -X POST "$API_URL/api/v1/inspect" \
      -H "Content-Type: application/json" \
      -d "{
        \"scan_id\": \"TEST-001\",
        \"image_base64\": \"$IMAGE_BASE64\",
        \"declared_items\": [\"clothing\", \"electronics\"]
      }" | python -m json.tool
else
    echo "Skipped: No test image available"
    echo ""
    echo "To test with your own image:"
    echo "  curl -X POST $API_URL/api/v1/inspect \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{"
    echo "      \"scan_id\": \"TEST-001\","
    echo "      \"image_base64\": \"<base64_encoded_image>\","
    echo "      \"declared_items\": [\"clothing\", \"electronics\"]"
    echo "    }'"
fi

echo ""
echo ""
echo "Test 3: API Documentation"
echo "========================="
echo "Open in browser: $API_URL/docs"

echo ""
echo "=========================================="
echo "API Testing Complete!"
echo "=========================================="
