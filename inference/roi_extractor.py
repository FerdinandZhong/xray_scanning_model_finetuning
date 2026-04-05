#!/usr/bin/env python3
"""
Extract ROI (Region of Interest) crops from X-ray images using YOLO proposals.

Supports both Ultralytics YOLO (.pt) and a fallback grid-based proposal method
for when no YOLO model is available.
"""

import json
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


def bbox_to_location(x_center: float, y_center: float) -> str:
    """Convert normalized center coordinates to location string."""
    h = "left" if x_center < 0.33 else ("right" if x_center > 0.67 else "center")
    v = "upper" if y_center < 0.33 else ("lower" if y_center > 0.67 else "center")
    if v == "center" and h == "center":
        return "center"
    if v == "center":
        return h
    if h == "center":
        return v
    return f"{v}-{h}"


class YOLOProposalGenerator:
    """Generate object proposals using a YOLO model."""

    def __init__(self, model_path: str, conf_threshold: float = 0.15, iou_threshold: float = 0.45):
        if not HAS_ULTRALYTICS:
            raise ImportError("ultralytics is required: pip install ultralytics")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = self.model.names

    def extract_rois(self, image_path: str, padding: float = 0.05) -> list[dict]:
        """
        Run YOLO on an image and extract ROI crops.

        Args:
            image_path: Path to the image.
            padding: Fraction of bbox size to add as padding around each crop.

        Returns:
            List of dicts with keys: crop (PIL Image), bbox, class_name,
            confidence, location, bbox_pixels.
        """
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        results = self.model.predict(
            image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False
        )

        rois = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x_c, y_c, w, h = box.xywhn[0].tolist()

                # Pixel coordinates with padding
                pad_w = w * padding
                pad_h = h * padding
                x1 = max(0, int((x_c - w / 2 - pad_w) * img_w))
                y1 = max(0, int((y_c - h / 2 - pad_h) * img_h))
                x2 = min(img_w, int((x_c + w / 2 + pad_w) * img_w))
                y2 = min(img_h, int((y_c + h / 2 + pad_h) * img_h))

                crop = image.crop((x1, y1, x2, y2))

                rois.append({
                    "crop": crop,
                    "bbox": [x_c, y_c, w, h],
                    "bbox_pixels": [x1, y1, x2, y2],
                    "class_name": self.class_names[cls_id],
                    "confidence": conf,
                    "location": bbox_to_location(x_c, y_c),
                })

        return rois


class YOLOAPIProposalGenerator:
    """Generate object proposals via a remote YOLO API endpoint."""

    def __init__(self, endpoint: str, conf_threshold: float = 0.15):
        self.endpoint = endpoint
        self.conf_threshold = conf_threshold

    def extract_rois(self, image_path: str, padding: float = 0.05) -> list[dict]:
        """Call YOLO API and extract ROI crops from detections."""
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        with open(image_path, "rb") as f:
            resp = requests.post(
                self.endpoint,
                files={"file": (Path(image_path).name, f, "image/jpeg")},
                headers={"accept": "application/json"},
                timeout=30,
            )
        resp.raise_for_status()
        data = resp.json()

        rois = []
        for item in data.get("items", []):
            conf = item.get("confidence", 0)
            if conf < self.conf_threshold:
                continue

            bbox = item.get("bbox", [0, 0, 0, 0])
            x_c, y_c, w, h = bbox

            pad_w = w * padding
            pad_h = h * padding
            x1 = max(0, int((x_c - w / 2 - pad_w) * img_w))
            y1 = max(0, int((y_c - h / 2 - pad_h) * img_h))
            x2 = min(img_w, int((x_c + w / 2 + pad_w) * img_w))
            y2 = min(img_h, int((y_c + h / 2 + pad_h) * img_h))

            crop = image.crop((x1, y1, x2, y2))

            rois.append({
                "crop": crop,
                "bbox": bbox,
                "bbox_pixels": [x1, y1, x2, y2],
                "class_name": item.get("name", "unknown"),
                "confidence": conf,
                "location": item.get("location", bbox_to_location(x_c, y_c)),
            })

        return rois


class GridProposalGenerator:
    """Fallback: generate grid-based proposals when no YOLO model is available."""

    def __init__(self, grid_size: int = 3, overlap: float = 0.25):
        self.grid_size = grid_size
        self.overlap = overlap

    def extract_rois(self, image_path: str, **kwargs) -> list[dict]:
        """Generate overlapping grid crops from the image."""
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        cell_w = img_w / self.grid_size
        cell_h = img_h / self.grid_size
        pad_w = cell_w * self.overlap
        pad_h = cell_h * self.overlap

        rois = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x1 = max(0, int(col * cell_w - pad_w))
                y1 = max(0, int(row * cell_h - pad_h))
                x2 = min(img_w, int((col + 1) * cell_w + pad_w))
                y2 = min(img_h, int((row + 1) * cell_h + pad_h))

                crop = image.crop((x1, y1, x2, y2))
                x_c = (x1 + x2) / 2 / img_w
                y_c = (y1 + y2) / 2 / img_h

                rois.append({
                    "crop": crop,
                    "bbox": [x_c, y_c, (x2 - x1) / img_w, (y2 - y1) / img_h],
                    "bbox_pixels": [x1, y1, x2, y2],
                    "class_name": "region",
                    "confidence": 1.0,
                    "location": bbox_to_location(x_c, y_c),
                })

        return rois


def create_proposal_generator(
    model_path: Optional[str] = None,
    endpoint: Optional[str] = None,
    **kwargs,
):
    """Factory: return YOLO API, local YOLO, or grid fallback."""
    if endpoint:
        api_kwargs = {k: v for k, v in kwargs.items() if k in ("conf_threshold",)}
        return YOLOAPIProposalGenerator(endpoint, **api_kwargs)
    if model_path and Path(model_path).exists() and HAS_ULTRALYTICS:
        return YOLOProposalGenerator(model_path, **kwargs)
    # Grid fallback doesn't use YOLO-specific kwargs
    grid_kwargs = {k: v for k, v in kwargs.items() if k in ("grid_size", "overlap")}
    return GridProposalGenerator(**grid_kwargs)
