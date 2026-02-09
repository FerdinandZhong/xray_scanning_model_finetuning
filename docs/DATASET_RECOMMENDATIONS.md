# Dataset Recommendations for X-ray Baggage Inspection

## Problem with OPIXray

While OPIXray is a good starting point, it has significant limitations for real-world customs inspection:

### Limitations
1. **Only 5 categories** (all knife-types)
2. **Few multi-object images** (~10% have 2+ items)
3. **No common items** (phones, laptops, bottles, clothing)
4. **Small dataset** (8,885 images)
5. **Limited scenarios** (single prohibited item per scan)

### Impact
- VLM won't recognize common baggage items
- Poor performance on cluttered, real-world scans
- Can't handle multiple object types simultaneously
- Limited generalization capability

## Recommended Datasets

### 1. SIXray (RECOMMENDED) ⭐

**Best for production use**

**Stats:**
- **1,059,231 X-ray images** (huge!)
- **6 categories**: Gun, Knife, Wrench, Pliers, Scissors, Hammer
- **Multiple objects per image**: Yes
- **Real baggage context**: Yes, includes common items

**Advantages:**
- ✅ Large-scale (100x bigger than OPIXray)
- ✅ More diverse prohibited items
- ✅ Real cluttered baggage scenarios
- ✅ Multiple objects per image
- ✅ Better for VLM training

**Download:**
- GitHub: https://github.com/MeioJane/SIXray
- Paper: https://arxiv.org/abs/1901.00303

**Categories:**
```python
SIXRAY_CATEGORIES = {
    "Gun": ["pistol", "rifle", "handgun"],
    "Knife": ["knife", "blade", "dagger"],
    "Wrench": ["wrench", "spanner"],
    "Pliers": ["pliers", "tongs"],
    "Scissors": ["scissors"],
    "Hammer": ["hammer"]
}
```

### 2. GDXray (Good Alternative)

**Stats:**
- **19,407 X-ray images**
- **Multiple domains**: Castings, welds, baggage, nature
- **Defect detection**: Yes

**Advantages:**
- ✅ More diverse than OPIXray
- ✅ Industrial quality control scenarios
- ✅ Good for transfer learning

**Disadvantages:**
- ❌ Smaller than SIXray
- ❌ Not focused on security

**Download:**
- Website: https://domingomery.ing.puc.cl/material/gdxray/

### 3. HiXray (Very Comprehensive) ⭐⭐

**Best for comprehensive coverage**

**Stats:**
- **45,364 X-ray images**
- **8 categories**: Portable_Charger_1, Portable_Charger_2, Mobile_Phone, Laptop, Tablet, Cosmetic, Water, Nonmetallic_Lighter
- **Real security checkpoint data**
- **Multiple objects**: Yes, realistic baggage

**Advantages:**
- ✅ Includes common items (phones, laptops, cosmetics)
- ✅ Real security scenarios
- ✅ Multiple objects per scan
- ✅ High resolution
- ✅ Modern items (power banks, tablets)

**Download:**
- Paper: https://arxiv.org/abs/2206.08661
- Contact authors for dataset access

### 4. COMPASS-XP (Future-Ready)

**Most comprehensive, if available**

**Stats:**
- Combines multiple X-ray datasets
- Standardized format
- Cross-domain validation

## Recommended Strategy 🎯

### Phase 1: Quick Start (Current)
```
Use: OPIXray
Purpose: Proof of concept, validate pipeline
Duration: 1-2 weeks
Expected Accuracy: 75-80% (limited categories)
```

### Phase 2: Production Dataset
```
Use: SIXray (primary) + HiXray (if available)
Purpose: Production-ready model
Duration: 2-4 weeks
Expected Accuracy: 85-90%
```

### Phase 3: Domain Adaptation
```
Use: Real JKDM X-ray data
Purpose: Fine-tune for specific scanner types and local context
Duration: Ongoing
Expected Accuracy: 90-95%
```

## Multi-Dataset Training Strategy

### Option A: Sequential Fine-tuning
```python
# Step 1: Pre-train on SIXray (large dataset)
python training/train_local.py --config configs/train_sixray.yaml

# Step 2: Fine-tune on OPIXray (occlusion focus)
python training/train_local.py \
  --config configs/train_opixray.yaml \
  --resume-from-checkpoint outputs/sixray_model

# Step 3: Fine-tune on real JKDM data
python training/train_local.py \
  --config configs/train_jkdm.yaml \
  --resume-from-checkpoint outputs/opixray_model
```

### Option B: Mixed Dataset Training
```python
# Combine datasets
python data/merge_datasets.py \
  --datasets sixray,opixray,hixray \
  --output data/combined_vqa.jsonl

# Train on combined data
python training/train_local.py --config configs/train_combined.yaml
```

## Creating a Comprehensive VQA Dataset

### Expand Categories

Instead of just 5 categories, aim for 20-50 categories:

**Prohibited Items:**
- Weapons: guns, knives, explosives, tasers
- Drugs: packages, powders, liquids
- Sharp objects: scissors, blades, tools
- Flammable: lighters, aerosols, chemicals

**Common Items (for context):**
- Electronics: phones, laptops, tablets, chargers
- Liquids: water bottles, cosmetics, toiletries
- Clothing: shoes, belts, bags
- Food: snacks, beverages
- Personal items: keys, wallets, jewelry

### Multi-Object Training Examples

```json
{
  "question": "What items are visible in this X-ray scan?",
  "answer": "Detected items: laptop, mobile phone, water bottle, folding knife (prohibited).",
  "metadata": {
    "categories": ["Laptop", "Mobile_Phone", "Water_Bottle", "Folding_Knife"],
    "prohibited_items": ["Folding_Knife"],
    "common_items": ["Laptop", "Mobile_Phone", "Water_Bottle"]
  }
}
```

## Implementation Plan

### Week 1-2: OPIXray Prototype
```bash
# Current implementation - DONE
python data/create_vqa_pairs.py --opixray-root data/opixray
python training/train_local.py --config configs/train_local.yaml
```

### Week 3-4: SIXray Integration
```bash
# Download SIXray
python data/download_sixray.py --output-dir data/sixray

# Convert to VQA format
python data/create_vqa_pairs.py \
  --dataset sixray \
  --root data/sixray \
  --output data/sixray_vqa.jsonl

# Train
python training/train_local.py --config configs/train_sixray.yaml
```

### Week 5-6: HiXray Integration (if available)
```bash
# Similar process for HiXray
python data/download_hixray.py --output-dir data/hixray
python data/create_vqa_pairs.py --dataset hixray --root data/hixray
```

### Week 7-8: Multi-Dataset Training
```bash
# Combine and train
python data/merge_datasets.py --output data/combined_vqa.jsonl
python training/train_local.py --config configs/train_combined.yaml
```

## Expected Performance Improvements

| Dataset | Images | Categories | Multi-Object | Expected Accuracy |
|---------|--------|------------|--------------|-------------------|
| OPIXray only | 8,885 | 5 | ~10% | 75-80% |
| SIXray | 1M+ | 6 | ~40% | 85-88% |
| SIXray + OPIXray | 1.05M | 11 | ~40% | 87-90% |
| SIXray + HiXray | 1.1M | 14+ | ~60% | 88-92% |
| + Real JKDM data | 1.1M+ | 20+ | ~70% | 90-95% |

## Cost-Benefit Analysis

### OPIXray (Current)
- **Cost**: Low (small dataset, quick training)
- **Benefit**: Proof of concept, validate pipeline
- **Limitation**: Not production-ready
- **Use**: Phase 1 only

### SIXray
- **Cost**: Medium (larger dataset, more training time)
- **Benefit**: Production-quality multi-object detection
- **Limitation**: Still limited categories
- **Use**: Phase 2 production

### Real JKDM Data Collection
- **Cost**: High (data collection, labeling, privacy)
- **Benefit**: Domain-specific, best accuracy
- **Limitation**: Time-consuming
- **Use**: Phase 3 optimization

## Recommendations 🎯

### For Proof of Concept (Current)
**Continue with OPIXray**
- ✅ Validates the pipeline works
- ✅ Quick iteration
- ✅ Tests VLM + post-processing architecture
- ⚠️ Accept limited accuracy (75-80%)

### For Production Deployment
**Migrate to SIXray**
- ✅ 100x more data
- ✅ Better multi-object scenarios
- ✅ More diverse prohibited items
- ✅ Production-ready accuracy (85-90%)

### For Best Results
**Multi-dataset approach**
- Start with SIXray (large-scale pre-training)
- Add OPIXray (occlusion specialization)
- Add HiXray (common items)
- Fine-tune with real JKDM data (domain adaptation)

## Next Steps

1. **Immediate (Week 1-2)**
   - Complete OPIXray proof of concept
   - Validate architecture and pipeline
   - Measure baseline performance

2. **Short-term (Week 3-6)**
   - Download SIXray dataset
   - Create adapter for SIXray → VQA format
   - Train production model

3. **Medium-term (Week 7-12)**
   - Integrate HiXray if available
   - Multi-dataset training
   - Deploy production model

4. **Long-term (Month 4+)**
   - Collect real JKDM X-ray data
   - Domain adaptation
   - Continuous improvement with feedback loop

## Conclusion

**OPIXray is suitable for:**
- ✅ Proof of concept
- ✅ Pipeline validation
- ✅ Architecture testing

**OPIXray is NOT suitable for:**
- ❌ Production deployment
- ❌ Multi-object recognition
- ❌ Diverse item detection

**Action:** Use OPIXray now, plan migration to SIXray for production.
