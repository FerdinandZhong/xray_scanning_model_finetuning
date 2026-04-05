# Dataset Plan: All-Items Detection for Customs Declaration Verification

## Objective

Build a VLM that recognizes **all visible items** in X-ray baggage scans (not just threats), maps them to **customs declaration categories**, and flags mismatches against passenger declarations.

---

## Dataset Inventory

### Already in Repository

| Dataset | Images | Item Categories | Everyday Items | Threat Items |
|---------|--------|----------------|----------------|--------------|
| **STCray** | 46,642 | 22 | 0 (only "Non Threat" meta-label) | 21 |
| **Luggage X-ray** | 7,120 | 12 | 7 (beverages, containers) | 5 |
| **CargoXray** | 659 | 16 | 16 (clothes, shoes, toys, etc.) | 0 |
| **X-Ray Baggage** | 1,500 | 5 | 0 | 5 |

### To Acquire

| Dataset | Images | Item Categories | Everyday Items | Threat Items | Download |
|---------|--------|----------------|----------------|--------------|----------|
| **HiXray** | 45,364 | 8 | 6 (phone, laptop, tablet, cosmetic, water, charger) | 2 | [Google Drive](https://drive.google.com/file/d/1jEk-h5Uv0-d3RdLf8cSHKXhuhalqD3l4/view) |
| **ORXray** | 10,933 | 10 | 4 (bottles, electronics, umbrella, battery) | 6 | Contact authors |

### Synthetic Labels (to generate)

| Source | Images | Method | New Categories |
|--------|--------|--------|----------------|
| **STCray (re-label)** | 46,642 | GPT-4o / Gemini labels ALL visible items | ~15-20 everyday categories |

---

## Phased Rollout

### Phase 0: Zero-Shot Baseline (no dataset needed)

- Test base Qwen3-VL-2B with "list all items" prompt on existing STCray test images
- No training, no new data
- Goal: establish baseline quality

### Phase 1: HiXray Integration

- **Download HiXray** (45K images, 8 categories)
- Convert to VQA format with "all items" prompt
- Provides: Laptop, Phone, Tablet, Charger, Cosmetic, Water
- **+6 new everyday item categories**

### Phase 2: Multi-Dataset Merge

- Combine HiXray + Luggage X-ray + CargoXray
- Unified category mapping
- Provides: beverages, containers, clothing, shoes, toys, office supplies
- **+19 new everyday item categories** (after dedup)

### Phase 3: Synthetic Labeling of STCray

- Use GPT-4o or Gemini to label ALL visible items in STCray images
- The 46K images already show everyday items -- they're just not annotated
- Generate VQA pairs: "What items are in this scan?" -> comprehensive JSON
- **+10-15 new everyday item categories** (clothing, bags, electronics, food, etc.)

### Phase 4: ORXray + Fine-Tuning

- Download ORXray (11K images, oriented bounding boxes)
- Adds bottles (glass/plastic/metal), electronics, umbrella
- Fine-tune on combined dataset for maximum coverage

---

## Complete Category Inventory

### Item-Level Categories (50 unique after dedup)

#### Threat/Prohibited Items (21 categories)
| # | Category | Source Dataset(s) |
|---|----------|------------------|
| 1 | Explosive | STCray |
| 2 | Gun / Firearm | STCray, ORXray, X-Ray Baggage |
| 3 | 3D Gun | STCray |
| 4 | Knife | STCray, Luggage X-ray, ORXray, X-Ray Baggage |
| 5 | Dagger | STCray, Luggage X-ray |
| 6 | Blade | STCray, Luggage X-ray |
| 7 | Scissors | STCray, Luggage X-ray, X-Ray Baggage |
| 8 | Swiss Army Knife | Luggage X-ray |
| 9 | Lighter | STCray, HiXray, ORXray |
| 10 | Injection / Syringe | STCray |
| 11 | Bullet / Ammunition | STCray |
| 12 | Handcuffs | STCray |
| 13 | Other Sharp Item | STCray |
| 14 | Nail Cutter | STCray |
| 15 | Hammer | STCray, X-Ray Baggage |
| 16 | Pliers | STCray, X-Ray Baggage |
| 17 | Wrench | STCray, X-Ray Baggage |
| 18 | Screwdriver | STCray |
| 19 | Pressure Vessel | ORXray |
| 20 | Spray Can (aerosol) | Luggage X-ray |
| 21 | Shaving Razor | STCray (YOLO test) |

#### Electronics (6 categories)
| # | Category | Source Dataset(s) |
|---|----------|------------------|
| 22 | Laptop | HiXray |
| 23 | Mobile Phone | HiXray |
| 24 | Tablet | HiXray |
| 25 | Portable Charger / Powerbank | HiXray, STCray |
| 26 | Battery | STCray, ORXray |
| 27 | Electronic Equipment (general) | ORXray |

#### Liquids & Containers (7 categories)
| # | Category | Source Dataset(s) |
|---|----------|------------------|
| 28 | Water / Water Bottle | HiXray |
| 29 | Glass Bottle | Luggage X-ray, ORXray |
| 30 | Plastic Bottle | Luggage X-ray, ORXray |
| 31 | Metal Bottle | ORXray |
| 32 | Cans | Luggage X-ray |
| 33 | Carton Drinks | Luggage X-ray |
| 34 | Vacuum Cup / Thermos | Luggage X-ray |

#### Cosmetics & Toiletries (1 category)
| # | Category | Source Dataset(s) |
|---|----------|------------------|
| 35 | Cosmetic / Toiletries | HiXray |

#### Clothing & Textiles (4 categories)
| # | Category | Source Dataset(s) |
|---|----------|------------------|
| 36 | Clothes | CargoXray |
| 37 | Shoes | CargoXray |
| 38 | Fabrics / Textiles | CargoXray |
| 39 | Bags | CargoXray |

#### Household & Personal (7 categories)
| # | Category | Source Dataset(s) |
|---|----------|------------------|
| 40 | Tableware | CargoXray |
| 41 | Toys | CargoXray |
| 42 | Lamps | CargoXray |
| 43 | Office Supplies | CargoXray |
| 44 | Umbrella | ORXray |
| 45 | Tin (containers) | Luggage X-ray |
| 46 | Tools (general) | CargoXray |

#### Synthetic Labels (estimated, from Phase 3)
| # | Category | Source |
|---|----------|--------|
| 47 | Food / Snacks | GPT-4o labeling of STCray |
| 48 | Books / Documents | GPT-4o labeling of STCray |
| 49 | Keys / Wallet / Jewelry | GPT-4o labeling of STCray |
| 50 | Medication | GPT-4o labeling of STCray |

---

## Customs Declaration Category Mapping

The 50 item-level categories map to **12 standard customs categories**:

| # | Customs Category | Mapped Item Categories | Declaration Checkbox |
|---|-----------------|----------------------|---------------------|
| 1 | **Electronics** | Laptop, Phone, Tablet, Charger, Powerbank, Battery, Electronic Equipment | "Carrying electronics?" |
| 2 | **Clothing / Textiles** | Clothes, Shoes, Fabrics, Textiles, Bags | "Carrying clothing/textiles?" |
| 3 | **Liquids** | Water, Glass/Plastic/Metal Bottle, Spray Can, Pressure Vessel | "Carrying liquids?" |
| 4 | **Food / Beverages** | Cans, Carton Drinks, Food, Snacks | "Carrying food items?" |
| 5 | **Toiletries / Cosmetics** | Cosmetic, Toiletries | "Carrying cosmetics?" |
| 6 | **Metal Tools** | Hammer, Pliers, Wrench, Screwdriver, Tools | "Carrying tools?" |
| 7 | **Weapons / Ammunition** | Gun, 3D Gun, Explosive, Bullet, Handcuffs | "Carrying weapons?" |
| 8 | **Sharp Objects** | Knife, Dagger, Blade, Scissors, Swiss Army Knife, Shaving Razor, Nail Cutter, Other Sharp Item | "Carrying sharp objects?" |
| 9 | **Flammable Items** | Lighter, Spray Can (aerosol) | "Carrying flammable items?" |
| 10 | **Documents / Books** | Office Supplies, Books, Documents | "Carrying documents?" |
| 11 | **Medication** | Injection/Syringe, Medication | "Carrying medication?" |
| 12 | **Other / Miscellaneous** | Umbrella, Tableware, Toys, Lamps, Vacuum Cup, Tin, Keys, Wallet, Jewelry | "Other items to declare?" |

---

## Summary

| Metric | Count |
|--------|-------|
| **Item-level categories (unique)** | **50** |
| **Customs declaration categories** | **12** |
| **Total images (all datasets)** | **~112,000** |
| **Datasets used** | 6 existing + synthetic labels |
| **New datasets to download** | 2 (HiXray, ORXray) |

### Coverage by Phase

| Phase | Cumulative Item Categories | Cumulative Customs Categories | Images Available |
|-------|---------------------------|------------------------------|-----------------|
| Phase 0 (zero-shot) | 0 (open vocab) | 12 (via VLM general knowledge) | 0 (no training) |
| Phase 1 (+HiXray) | 28 | 8 | 52,484 |
| Phase 2 (+Luggage+Cargo) | 46 | 11 | 60,263 |
| Phase 3 (+STCray synthetic) | 50 | 12 | 106,905 |
| Phase 4 (+ORXray) | 50 | 12 | 117,838 |
