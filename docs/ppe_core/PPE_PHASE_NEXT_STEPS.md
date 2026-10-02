# PS6 PPE-Core Phase — Immediate Execution Order

1. Download `51ddhesh/PPE_Detection` and preserve it unchanged.
2. Filter to helmet/gloves/goggles/vest.
3. Deduplicate across splits.
4. Validate + visually inspect annotations.
5. Train YOLO26n PPE detector on GPU.
6. Evaluate per class on unseen test data.
7. Freeze the best PPE checkpoint.
8. Run multi-worker video with separate person tracking.
9. Add worker↔PPE association.
10. Add compliance rule: all four required.
11. Add 3-frame persistence / 2-frame clear logic.
12. Produce compliant and non-compliant demo cases.
13. Only then wire the result into the later backend/dashboard.
