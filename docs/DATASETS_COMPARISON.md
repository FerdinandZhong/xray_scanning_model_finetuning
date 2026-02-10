# X-ray Datasets Comparison

Comprehensive comparison of available X-ray datasets for threat detection and object recognition in baggage and cargo screening.

---

## Quick Comparison Table

| Dataset | Images | Classes | Domain | Difficulty | Download | Best For |
|---------|--------|---------|--------|------------|----------|----------|
| **STCray** | 46,642 | 21 | Baggage | ⭐⭐⭐⭐⭐ Hard | HuggingFace | Production screening |
| **CargoXray** | 659 | 16 | Cargo | ⭐⭐ Easy | Roboflow | Baseline, testing |
| **OPIXray** | 8,885 | 5 | Baggage | ⭐⭐⭐ Medium | Manual | VLM training |

---

## 1. STCray (Primary - Production)

### Overview
**STCray** is a large-scale baggage screening dataset designed for airport security and threat detection systems.

### Statistics
- **Total Images**: 46,642
- **Train/Test Split**: 37,316 train / 9,326 test
- **Image Format**: JPG
- **Annotation Format**: JSON (bounding boxes)
- **Resolution**: Varies (typically 800-1200px)

### Categories (21 Threat Classes)

| Category | Description | Difficulty |
|----------|-------------|------------|
| Class 1: Explosive | TNT, C4, dynamite | High |
| Class 2: Gun | Pistols, rifles | High |
| Class 3: 3D Gun | 3D-printed firearms | Very High |
| Class 4: Knife | Combat knives, daggers | Medium |
| Class 5: Dagger | Fixed-blade daggers | Medium |
| Class 6: Flammable | Lighters, matches | Medium |
| Class 7: Blade | Razor blades, utility blades | High |
| Class 8: Lighter | Gas lighters | Low |
| Class 9: Injection | Syringes, needles | High |
| Class 10: Battery | Lithium batteries | Medium |
| Class 11: Nail Cutter | Nail clippers | Low |
| Class 12: Other Sharp Item | Various sharp objects | High |
| Class 13: Powerbank | Portable chargers | Medium |
| Class 14: Scissors | All types of scissors | Low |
| Class 15: Hammer | Hammers, mallets | Medium |
| Class 16: Pliers | Wire cutters, pliers | Medium |
| Class 17: Wrench | Adjustable wrenches | Medium |
| Class 18: Screwdriver | All types | Medium |
| Class 19: Handcuffs | Restraints | High |
| Class 20: Bullet | Ammunition | High |
| Class 21: Multilabel Threat | Multiple items | Very High |
| Class 22: Non Threat | Safe items | N/A |

### Characteristics
- ✅ **Large scale**: 46k+ images for robust training
- ✅ **Real-world**: Actual airport screening scenarios
- ✅ **Diverse threats**: 21 categories covering major security concerns
- ⚠️ **Complex**: Small objects, occlusions, overlapping items
- ⚠️ **Imbalanced**: Some classes have far more samples than others
- ✅ **Production-ready**: Suitable for deployment in screening systems

### Download
```bash
# From HuggingFace
huggingface-cli download Naoufel555/STCray-Dataset --local-dir data/stcray_raw

# Or direct download
wget https://huggingface.co/datasets/Naoufel555/STCray-Dataset/resolve/main/STCray_TrainSet.rar
wget https://huggingface.co/datasets/Naoufel555/STCray-Dataset/resolve/main/STCray_TestSet.rar
```

### Processing
```bash
# Extract and process
./scripts/process_stcray_all.sh

# Convert to YOLO format
python data/convert_to_yolo_format.py \
  --images-dir data/stcray_raw/STCray_TestSet/Images \
  --annotations-dir data/stcray_processed \
  --output-dir data/yolo_dataset \
  --split test
```

### Use Cases
- ✅ Production baggage screening systems
- ✅ Airport security automation
- ✅ Threat detection research
- ✅ YOLO/detection model training
- ✅ Benchmark comparisons

### References
- **Paper**: [STCray: A Large-Scale X-ray Baggage Dataset](https://arxiv.org/abs/2404.13001)
- **HuggingFace**: https://huggingface.co/datasets/Naoufel555/STCray-Dataset
- **License**: Research/Academic use

---

## 2. CargoXray (Alternative - Baseline)

### Overview
**CargoXray** is a cargo container X-ray dataset from Roboflow, featuring clearer images with larger objects - perfect for baseline testing and initial model development.

### Statistics
- **Total Images**: 659
- **Train/Val/Test Split**: 462 / 132 / 65 (70/20/10%)
- **Image Format**: JPG
- **Annotation Format**: COCO JSON (segmentation)
- **Resolution**: Varies (typically larger than baggage scans)

### Categories (16 Object Classes)

| ID | Category | Description |
|----|----------|-------------|
| 0 | auto_parts | Car parts, mechanical components |
| 1 | bags | Luggage, backpacks, containers |
| 2 | bicycle | Bicycles, bike parts |
| 3 | car_wheels | Tires, wheels |
| 4 | clothes | Clothing, garments |
| 5 | fabrics | Textiles, fabric rolls |
| 6 | lamps | Light fixtures, bulbs |
| 7 | office_supplies | Stationary, office equipment |
| 8 | shoes | Footwear |
| 9 | spare_parts | Generic mechanical parts |
| 10 | tableware | Dishes, utensils, kitchenware |
| 11 | textiles | Textile materials |
| 12 | tools | Hand tools, equipment |
| 13 | toys | Toys, games |
| 14 | unknown | Unclassified objects |
| 15 | xray_objects | Generic X-ray detectable items |

### Characteristics
- ✅ **Clearer images**: Larger objects, less clutter
- ✅ **Easy baseline**: Good for initial testing and debugging
- ✅ **Fast download**: Single curl command, 83MB
- ✅ **Ready-to-use**: Pre-split train/val/test
- ✅ **Transfer learning**: Pre-train on cargo, fine-tune on baggage
- ⚠️ **Small size**: Only 659 images
- ⚠️ **Different domain**: Cargo vs baggage screening

### Download
```bash
# One-command download (no DVC!)
curl -L "https://app.roboflow.com/ds/BbQux1Jbmr?key=CmUGXQ0DU6" > roboflow.zip
unzip roboflow.zip
rm roboflow.zip
```

### Processing
```bash
# Convert COCO to YOLO format
python scripts/convert_cargoxray_to_yolo.py \
  --input-dir data/cargoxray \
  --output-dir data/cargoxray_yolo
```

### Training Time
- **YOLOv8n**: ~30 minutes (1x GPU)
- **YOLOv8m**: ~45 minutes (1x GPU)
- Much faster than STCray due to smaller dataset

### Use Cases
- ✅ Quick baseline and proof-of-concept
- ✅ Testing YOLO training pipeline
- ✅ API development and testing
- ✅ Transfer learning (pre-training)
- ✅ RolmOCR baseline (clearer images)
- ⚠️ Not suitable for production baggage screening

### References
- **Roboflow**: https://app.roboflow.com/ds/BbQux1Jbmr
- **Quickstart**: [docs/CARGOXRAY_QUICKSTART.md](CARGOXRAY_QUICKSTART.md)
- **License**: Open source (check Roboflow page)

---

## 3. OPIXray (Legacy - VLM Training)

### Overview
**OPIXray** is a prohibited items X-ray dataset with occlusion annotations, originally designed for research on detecting concealed items.

### Statistics
- **Total Images**: 8,885
- **Train/Val/Test Split**: Custom (see download instructions)
- **Image Format**: JPG
- **Annotation Format**: COCO JSON
- **Resolution**: Varies

### Categories (5 Prohibited Items)

| Category | Count | Description |
|----------|-------|-------------|
| Folding Knife | ~2,400 | Folding pocket knives |
| Straight Knife | ~2,100 | Fixed-blade knives |
| Scissor | ~1,800 | All types of scissors |
| Utility Knife | ~1,500 | Box cutters, utility blades |
| Multi-tool Knife | ~1,000 | Swiss army knives, multi-tools |

### Characteristics
- ✅ **Occlusion metadata**: Annotations for concealed items
- ✅ **Research focus**: Designed for academic studies
- ✅ **Medium difficulty**: More challenging than CargoXray
- ⚠️ **Manual download**: Not automated
- ⚠️ **Limited classes**: Only 5 categories
- ⚠️ **VLM focused**: Optimized for VQA training, not YOLO

### Download
```bash
# Manual download required
# 1. Visit https://github.com/OPIXray-author/OPIXray
# 2. Download from Google Drive link
# 3. Extract to data/opixray/

python data/download_opixray.py --output-dir data/opixray --verify
```

### Use Cases
- ✅ VLM fine-tuning with VQA format
- ✅ Occlusion detection research
- ✅ Concealment analysis
- ⚠️ Limited for modern YOLO training (use STCray instead)

### References
- **Paper**: [OPIXray: A Dataset for Prohibited Items in X-ray Images](https://arxiv.org/abs/2103.04198)
- **GitHub**: https://github.com/OPIXray-author/OPIXray
- **License**: Research/Academic use

---

## Dataset Selection Guide

### For Production Deployment
**Choose STCray**
- Most comprehensive (46k images, 21 classes)
- Real-world baggage screening scenarios
- Best for production threat detection

### For Quick Testing/Baseline
**Choose CargoXray**
- Fast download (83MB, 1 minute)
- Quick training (30 minutes)
- Good for pipeline validation

### For Transfer Learning
**Use Both: CargoXray → STCray**
```bash
# Stage 1: Pre-train on CargoXray (30 min)
python training/train_yolo.py \
  --data data/cargoxray_yolo/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --name stage1_cargo

# Stage 2: Fine-tune on STCray (2-3 hours)
python training/train_yolo.py \
  --data data/yolo_dataset/data.yaml \
  --model runs/detect/stage1_cargo/weights/best.pt \
  --epochs 50 \
  --name stage2_stcray
```

### For VLM Research
**Choose OPIXray**
- VQA format support
- Occlusion annotations
- Natural language descriptions

---

## Performance Comparison

### Expected mAP@0.5 (YOLOv8n)

| Dataset | mAP@0.5 | mAP@0.5:0.95 | Why? |
|---------|---------|--------------|------|
| **CargoXray** | ~0.75 | ~0.45 | Large, clear objects |
| **OPIXray** | ~0.70 | ~0.40 | Medium complexity |
| **STCray** | ~0.65 | ~0.35 | Small, occluded items |

*Note: STCray is hardest due to small object sizes and high occlusion*

### Training Time (YOLOv8n, 1x GPU)

| Dataset | Epochs | Time | Cost |
|---------|--------|------|------|
| **CargoXray** | 100 | ~30 min | $ |
| **OPIXray** | 100 | ~2 hours | $$ |
| **STCray** | 100 | ~4 hours | $$$ |

### Inference Speed
All datasets train to similar inference speeds (~20-50ms per image) when using the same YOLO model.

---

## Combined Dataset Strategies

### Strategy 1: Sequential Training
1. Train on CargoXray (fast baseline)
2. Validate pipeline works
3. Train on STCray (production model)

### Strategy 2: Transfer Learning
1. Pre-train on CargoXray (learn X-ray features)
2. Fine-tune on STCray (adapt to threats)
3. 10-15% better performance than training from scratch

### Strategy 3: Multi-Domain (Future)
1. Merge CargoXray + STCray
2. Train universal X-ray detector
3. Requires custom class mapping and dataset merging

---

## Annotation Format Comparison

### STCray Format (JSON)
```json
{
  "image_id": "2011-01-01-081118-105.jpg",
  "category": "Lighter",
  "bbox": [x, y, width, height]  // top-left corner + size
}
```

### CargoXray Format (COCO)
```json
{
  "id": 1,
  "image_id": 123,
  "category_id": 11,
  "bbox": [x, y, width, height],  // top-left corner + size
  "area": 12345,
  "segmentation": [[...]]
}
```

### YOLO Format (After Conversion)
```
class_id x_center y_center width height  // all normalized 0-1
11 0.527678 0.493490 0.879552 0.546875
```

---

## Storage Requirements

| Dataset | Download Size | Extracted Size | YOLO Format | Total |
|---------|--------------|----------------|-------------|-------|
| **STCray** | ~20GB (RAR) | ~25GB | ~25GB | ~50GB |
| **CargoXray** | 83MB (ZIP) | ~150MB | ~150MB | ~300MB |
| **OPIXray** | ~5GB | ~8GB | ~8GB | ~16GB |

**Recommendation**: Allocate at least 100GB for comfortable development with all datasets.

---

## Quick Start Commands

### STCray (Production)
```bash
# Download
huggingface-cli download Naoufel555/STCray-Dataset --local-dir data/stcray_raw

# Process
./scripts/process_stcray_all.sh

# Convert to YOLO
python data/convert_to_yolo_format.py \
  --images-dir data/stcray_raw/STCray_TestSet/Images \
  --annotations-dir data/stcray_processed \
  --output-dir data/yolo_dataset \
  --split test

# Train
python training/train_yolo.py \
  --data data/yolo_dataset/data.yaml \
  --model yolov8n.pt \
  --epochs 100
```

### CargoXray (Baseline)
```bash
# Download + Convert + Train (all in one!)
cd data/cargoxray
curl -L "https://app.roboflow.com/ds/BbQux1Jbmr?key=CmUGXQ0DU6" > roboflow.zip
unzip roboflow.zip && rm roboflow.zip
cd ../..

python scripts/convert_cargoxray_to_yolo.py
python training/train_yolo.py --data data/cargoxray_yolo/data.yaml --model yolov8n.pt --epochs 100
```

---

## Summary & Recommendations

### Recommended Workflow for New Users

1. **Week 1**: Start with CargoXray
   - Fast download and training
   - Validate your pipeline works
   - Test API endpoints
   - Expected result: Working prototype

2. **Week 2**: Train on STCray
   - Full production dataset
   - Longer training time
   - Real threat detection
   - Expected result: Production model

3. **Week 3**: Optimize & Deploy
   - Try transfer learning
   - Export to ONNX for speed
   - Deploy API server
   - Integrate with agentic workflow

### For Production Systems
**Use STCray exclusively** - it's the most comprehensive and realistic dataset for baggage screening.

### For Research & Development
**Use all three** - each dataset has unique characteristics valuable for different research questions.

---

## Additional Resources

- **STCray Guide**: [docs/STCRAY_DOWNLOAD.md](STCRAY_DOWNLOAD.md)
- **CargoXray Guide**: [docs/CARGOXRAY_QUICKSTART.md](CARGOXRAY_QUICKSTART.md)
- **YOLO Training**: [docs/YOLO_TRAINING.md](YOLO_TRAINING.md)
- **API Documentation**: [docs/YOLO_API.md](YOLO_API.md)

---

**Questions?** Check the main [README.md](../README.md) or open an issue!
