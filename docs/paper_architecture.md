# XScan-Agent Paper Architecture Diagrams

## 1. System Architecture (End-to-End Pipeline)

```mermaid
flowchart TD
    INPUT["X-ray Image + Declaration Form"]

    subgraph PERCEPTION["1. Perception Layer"]
        YOLO["Class-Agnostic YOLO\n(localization only)"]
        PROPOSALS["Object Proposals\n[R1: upper-left, 0.85]\n[R2: center, 0.72]"]
        YOLO --> PROPOSALS
    end

    subgraph REASONING["2. Reasoning Core (Fine-tuned VLM)"]
        VLM_INPUT["Full Image\n+ YOLO Proposals as Spatial Tokens\n+ Category Hints (Phase C)"]
        VLM["Qwen3-VL-2B\n(PG-RAV Fine-tuned)"]
        VLM_OUTPUT["Per-Region Classification\n+ Token Logprobs"]
        VLM_INPUT --> VLM --> VLM_OUTPUT
    end

    subgraph ADAPTIVE["3. Adaptive Re-Analysis Loop"]
        ENTROPY{"Entropy >\nThreshold?"}
        ZOOM["Zoom ROI Crop"]
        HINTS["Lookup Confusable\nCategories"]
        REQUERY["Re-query VLM\n(Focused Format)"]
        UPDATE["Update if\nMore Confident"]
        ENTROPY -->|Yes| ZOOM --> HINTS --> REQUERY --> UPDATE
        ENTROPY -->|No| SKIP["Keep Original"]
    end

    subgraph COMPARATOR["4. Declaration Comparator"]
        CATMAP["Category Mapping\n(50+ items -> 12 customs)"]
        SETCMP["Set Comparison\nDetected vs Declared"]
        FLAG["Mismatch Flagging\n+ Severity Level"]
        CATMAP --> SETCMP --> FLAG
    end

    OUTPUT["Mismatch Report\n+ Item Inventory\n+ Evidence Crops\n+ Confidence Scores"]

    INPUT --> PERCEPTION
    PERCEPTION --> REASONING
    REASONING --> ADAPTIVE
    ADAPTIVE --> COMPARATOR
    COMPARATOR --> OUTPUT

    style PERCEPTION fill:#e1f5fe
    style REASONING fill:#fff3e0
    style ADAPTIVE fill:#f3e5f5
    style COMPARATOR fill:#e8f5e9
```

## 2. Fine-Tuning Curriculum (Three-Phase Training)

```mermaid
flowchart LR
    BASE["Qwen3-VL-2B\n(Pretrained)"]

    subgraph PHASE_A["Phase A: Domain Adaptation"]
        direction TB
        A_DATA["STCray Dataset\n46K images\n21 threat categories"]
        A_METHOD["QLoRA Fine-tuning\n4-bit NF4 + LoRA r=16"]
        A_RESULT["X-ray Adapted VLM\nUnderstands shapes,\nmaterials, overlaps"]
        A_DATA --> A_METHOD --> A_RESULT
    end

    subgraph PHASE_B["Phase B: Proposal-Guided Training"]
        direction TB
        B_YOLO["Class-Agnostic YOLO\nTrained on STCray+HiXray\n~90K images"]
        B_DUAL["Dual-Format Training"]
        B_F1["Format 1: Full-Scene\nProposals as spatial tokens\n-> Per-region classification"]
        B_F2["Format 2: Focused ROI\nCropped region + hints\n-> Single-item classification"]
        B_RESULT["Proposal-Guided VLM\nSpatially grounded\n50+ item categories"]
        B_YOLO --> B_DUAL
        B_DUAL --> B_F1
        B_DUAL --> B_F2
        B_F1 --> B_RESULT
        B_F2 --> B_RESULT
    end

    subgraph PHASE_C["Phase C: Retrieval-as-Text"]
        direction TB
        C_HINTS["Static Category Hints\n~50 entries\n{description, visual_cues,\nconfusables}"]
        C_METHOD["Inference-Time Only\nInject hints into prompt\nNo additional training"]
        C_RESULT["Disambiguated VLM\nReduced confusion\nbetween similar items"]
        C_HINTS --> C_METHOD --> C_RESULT
    end

    BASE --> PHASE_A
    PHASE_A --> PHASE_B
    PHASE_B --> PHASE_C

    style PHASE_A fill:#c8e6c9
    style PHASE_B fill:#bbdefb
    style PHASE_C fill:#f8bbd0
```

## 3. Ablation Structure (Paper Experiments)

```mermaid
flowchart TD
    subgraph ABLATION["Ablation Ladder"]
        E1["E1: YOLO Baseline\n(closed-set, 21 classes)"]
        E2["E2: Zero-shot VLM\n(no fine-tuning)"]
        E3["E3: + Phase A\n(domain adaptation)"]
        E4["E4: + Phase B\n(proposal-guided)"]
        E5["E5: + Phase C\n(category hints)"]
        E6["E6: + Adaptive Loop\n(uncertainty-gated)"]
        E7["E7: End-to-End\n(+ declaration matching)"]
    end

    subgraph BASELINES["External Baselines"]
        OWL["OWL-ViT\n(RGB open-vocab)"]
        DINO["Grounding DINO\n(RGB open-vocab)"]
        OVXD["OVXD\n(X-ray CLIP adaptation)"]
        RAXO["RAXO\n(training-free retrieval)"]
    end

    subgraph CALIBRATION["Calibration Experiments"]
        ECAL["E_cal: Logprob Calibration\nEntropy vs Accuracy\nTarget: ECE < 0.15"]
        EADAPT["E_adapt: Adaptive Loop\nSingle-pass vs Loop\nTarget: +5% recall"]
    end

    E1 --> E2 --> E3 --> E4 --> E5 --> E6 --> E7

    style ABLATION fill:#fff9c4
    style BASELINES fill:#ffccbc
    style CALIBRATION fill:#d1c4e9
```

## 4. Contributions Overview

```mermaid
flowchart TB
    subgraph C1["C1: Model Innovation"]
        C1_WHAT["Proposal-Guided Dual-Format\nVLM Fine-Tuning"]
        C1_NOVEL["Novel: YOLO proposals + category hints\nintegrated into VLM training,\nnot just inference"]
        C1_VS["vs OVXD: image features only\nvs RAXO: inference-time only\nvs crop-and-classify: loses context"]
    end

    subgraph C2["C2: System Innovation"]
        C2_WHAT["Uncertainty-Gated\nAdaptive Perception Loop"]
        C2_NOVEL["Novel: Logprob-driven re-analysis\nfeeds VLM the exact format\nit was trained on"]
        C2_VS["Phase C hints + orchestrator\n= training-aligned inference"]
    end

    subgraph C3["C3: Application"]
        C3_WHAT["End-to-End Agentic\nCustoms Verification"]
        C3_NOVEL["First VLM-based\ncustoms verification system"]
        C3_VS["Complete ablation\nshowing each component's\ncontribution"]
    end

    C1 --- C2 --- C3

    style C1 fill:#bbdefb
    style C2 fill:#f3e5f5
    style C3 fill:#c8e6c9
```

## 5. Comparison with Prior Work

```mermaid
flowchart LR
    subgraph PRIOR["Prior Work"]
        direction TB
        OVXD_M["OVXD\nCLIP adaptation\nImage features only\nNo spatial grounding"]
        RAXO_M["RAXO\nTraining-free retrieval\nRGB exemplars\nCross-domain gap"]
        CROP["Crop-and-Classify\nIndependent ROIs\nLoses full-image context"]
    end

    subgraph OURS["XScan-Agent (Ours)"]
        direction TB
        TRAIN["Training-time integration\nYOLO proposals + hints\nin VLM prompt"]
        INFER["Inference-time adaptation\nLogprob uncertainty\nAdaptive re-analysis"]
        E2E["End-to-end pipeline\nDeclaration verification\nAuditable decisions"]
    end

    OVXD_M -.->|"We add spatial\ngrounding"| TRAIN
    RAXO_M -.->|"We fine-tune WITH\nretrieval context"| TRAIN
    CROP -.->|"We keep full\nimage context"| INFER

    style PRIOR fill:#ffccbc
    style OURS fill:#c8e6c9
```
