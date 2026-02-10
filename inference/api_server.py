#!/usr/bin/env python3
"""
FastAPI server for X-ray inspection with vLLM backend.
Provides REST API for production deployment.
"""

import argparse
import base64
import io
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn

# Import inference engine and post-processing
from inference.vllm_engine import create_inference_engine
from inference.postprocess import process_vlm_response


# Request/Response models
class InspectionRequest(BaseModel):
    """Request model for X-ray inspection."""
    scan_id: str
    image_base64: Optional[str] = None
    declared_items: Optional[List[str]] = None
    question: Optional[str] = None
    mode: str = "vqa"  # vqa, detection, comparison
    use_structured: bool = True  # Use XGrammar structured JSON output


class DetectedItem(BaseModel):
    """Detected item in X-ray scan."""
    name: str
    confidence: float
    location: str
    occluded: bool = False


class InspectionResponse(BaseModel):
    """Response model for X-ray inspection."""
    scan_id: str
    risk_level: str  # low, medium, high
    detected_items: List[DetectedItem]
    item_details: List[Dict[str, Any]]  # Full details with confidence & location
    declaration_match: Optional[bool] = None
    reasoning: str
    recommended_action: str  # CLEAR, REVIEW, PHYSICAL_INSPECTION
    processing_time_ms: float
    used_structured_output: bool  # Whether XGrammar was used


# Initialize FastAPI app
app = FastAPI(
    title="X-ray Inspection API",
    description="AI-powered X-ray baggage inspection with VQA",
    version="1.0.0",
)

# Global inference engine instance
inference_engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup."""
    global inference_engine
    
    # Will be set via command line args
    print("FastAPI server started. Inference engine will be initialized via startup config.")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "X-ray Inspection API",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "engine_ready": inference_engine is not None,
        "timestamp": time.time(),
    }


@app.post("/api/v1/inspect", response_model=InspectionResponse)
async def inspect_scan(request: InspectionRequest):
    """
    Inspect X-ray scan and detect threats.
    
    Args:
        request: Inspection request with image and metadata
    
    Returns:
        Inspection results with risk assessment
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    start_time = time.perf_counter()
    
    try:
        # Decode image from base64
        if not request.image_base64:
            raise HTTPException(status_code=400, detail="image_base64 is required")
        
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save temporarily for inference
        temp_path = f"/tmp/scan_{request.scan_id}.jpg"
        image.save(temp_path)
        
        # Step 1: Run VQA inference with structured or natural output
        if request.use_structured:
            # Use XGrammar guided JSON generation
            if request.question:
                prompt = request.question
            else:
                prompt = "List all items detected in this X-ray scan in JSON format."
            
            vlm_answer = inference_engine.generate_structured(
                image_path=temp_path,
                prompt=prompt,
                max_tokens=500,
                temperature=0.7,
            )
        else:
            # Use natural language generation
            if request.question:
                prompt = request.question
            else:
                prompt = "What items are visible in this X-ray scan?"
            
            vlm_answer = inference_engine.generate_natural(
                image_path=temp_path,
                prompt=prompt,
                max_tokens=256,
                temperature=0.7,
            )
        
        # Step 2: Post-process VLM response with declaration comparison
        post_processed = process_vlm_response(
            vlm_answer=vlm_answer,
            declared_items=request.declared_items,
        )
        
        # Step 3: Format detected items for response
        detected_items_list = []
        for item_detail in post_processed.get("item_details", []):
            detected_items_list.append(DetectedItem(
                name=item_detail["name"],
                confidence=item_detail["confidence"],
                location=item_detail["location"],
                occluded=post_processed["has_occlusion"],
            ))
        
        # Step 4: Get declaration match info
        declaration_match = None
        if post_processed.get("declaration_comparison"):
            declaration_match = post_processed["declaration_comparison"]["declaration_match"]
        
        # Processing time
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000
        
        # Build response
        response = InspectionResponse(
            scan_id=request.scan_id,
            risk_level=post_processed["risk_level"],
            detected_items=detected_items_list,
            item_details=post_processed.get("item_details", []),
            declaration_match=declaration_match,
            reasoning=post_processed["reasoning"],
            recommended_action=post_processed["recommended_action"],
            processing_time_ms=processing_time_ms,
            used_structured_output=request.use_structured,
        )
        
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/v1/batch-inspect")
async def batch_inspect(requests: List[InspectionRequest]):
    """
    Batch inspection for multiple scans.
    
    Args:
        requests: List of inspection requests
    
    Returns:
        List of inspection results
    """
    if vllm_server is None:
        raise HTTPException(status_code=503, detail="vLLM server not initialized")
    
    results = []
    
    for req in requests:
        try:
            result = await inspect_scan(req)
            results.append(result)
        except Exception as e:
            results.append({
                "scan_id": req.scan_id,
                "error": str(e),
                "status": "failed",
            })
    
    return results


@app.post("/api/v1/upload-and-inspect")
async def upload_and_inspect(
    scan_id: str,
    file: UploadFile = File(...),
    declared_items: Optional[str] = None,
):
    """
    Upload image file and inspect.
    
    Args:
        scan_id: Scan identifier
        file: Uploaded image file
        declared_items: Comma-separated declared items
    
    Returns:
        Inspection result
    """
    # Read and encode image
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # Parse declared items
    items_list = None
    if declared_items:
        items_list = [item.strip() for item in declared_items.split(",")]
    
    # Create request
    request = InspectionRequest(
        scan_id=scan_id,
        image_base64=image_base64,
        declared_items=items_list,
    )
    
    # Inspect
    return await inspect_scan(request)


def main():
    parser = argparse.ArgumentParser(description="FastAPI server for X-ray inspection")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Uvicorn workers",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        default=True,
        help="Use vLLM engine (default, recommended for production)",
    )
    parser.add_argument(
        "--use-transformers",
        action="store_true",
        help="Use Transformers engine (fallback, no XGrammar)",
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    print("Initializing inference engine...")
    global inference_engine
    
    use_vllm = args.use_vllm and not args.use_transformers
    
    inference_engine = create_inference_engine(
        model_path=args.model,
        use_vllm=use_vllm,
        tensor_parallel_size=args.tensor_parallel_size if use_vllm else None,
        gpu_memory_utilization=args.gpu_memory_utilization if use_vllm else None,
    )
    print(f"✓ Inference engine ready ({'vLLM' if use_vllm else 'Transformers'})")
    
    print(f"\nStarting API server on {args.host}:{args.port}")
    print(f"API documentation: http://{args.host}:{args.port}/docs")
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
