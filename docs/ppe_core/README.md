# PS6 PPE Core Work Package

This package refocuses PS6 on the four mandatory PPE categories from the discussion slides:

- Helmet
- Gloves
- Goggles
- Safety vest

Fire and smoke are explicitly treated as optional extensions for now.

The package is designed to plug into the existing Factory-AI repository without replacing the already-validated M1 fire/smoke work.

## Core architecture

```text
CCTV / Video
    ↓
Person detection + tracking
    ↓
PPE detection (helmet / gloves / goggles / vest)
    ↓
Person ↔ PPE association
    ↓
Compliance engine
    ↓
Persistent-violation gate
    ↓
Explainable alert / dashboard
```

## Dataset strategy

Primary current candidate: `51ddhesh/PPE_Detection` on Hugging Face. Its dataset card declares CC BY 4.0 and six manually annotated PPE classes: Vest, Safety Shoe, Mask, Helmet, Goggles, Gloves. We retain only the four PS6 core classes and do not turn missing annotations into `no_*` detector classes.

Source: https://huggingface.co/datasets/51ddhesh/PPE_Detection

The source dataset does not include a person class, so the system keeps person detection separate using the existing COCO-pretrained YOLO26n person detector.

## Files

- `datasets/ppe_core.yaml` — target four-class training YAML
- `datasets/PPE_DATASET_STRATEGY.md` — dataset selection, mapping, leakage, and evaluation rules
- `scripts/prepare_ppe_core.py` — filters source YOLO labels to the four target classes and performs conservative duplicate quarantine
- `scripts/compliance_engine.py` — worker↔PPE association and persistent-violation logic
- `scripts/train_ppe.py` — GPU training entry point for Kaggle/Colab/RTX 4060
- `scripts/run_ppe_demo.py` — video inference demo using person tracking + PPE detection + compliance
- `ZAI_AGENT_PROMPT_PPE_CORE.txt` — copy-paste implementation prompt for Z.ai Agent Mode

## Important

This package does not claim that a model has already been trained on the new PPE dataset. Training must happen only after the dataset is downloaded, filtered, validated, and visually inspected.
