#!/usr/bin/env python3
"""
BMVC 2026 Research Paper Data Organizer
Auto-generates paper templates reflecting the DINOv2 + RoBERTa Silver Standard Architecture.
Includes Visual Learning Guides.
"""

import json
import shutil
import sys
from pathlib import Path


def create_research_paper_folder(training_output_dir, paper_output_dir="research_paper_data"):
    """
    Create organized research paper folder from training results.
    """
    paper_path = Path(paper_output_dir)
    paper_path.mkdir(parents=True, exist_ok=True)
    training_path = Path(training_output_dir)

    print("🚀 Organizing research paper data...")
    print(f"   Source: {training_path}")
    print(f"   Destination: {paper_path}\n")

    # 1. RESULTS TABLES
    print("📊 Organizing results tables...")
    results_dir = paper_path / "1_RESULTS_TABLES"
    results_dir.mkdir(exist_ok=True)

    files_to_copy = [
        ("RESEARCH_RESULTS_REPORT.pdf", "Report (professional PDF)"),
        ("RESULTS_TABLE.csv", "Data table (for Excel)"),
        ("RESULTS_LATEX_TABLE.txt", "LaTeX table (for Overleaf)"),
        ("summary.json", "All metrics (raw data)"),
    ]

    for filename, description in files_to_copy:
        src = training_path / filename
        if src.exists():
            shutil.copy2(src, results_dir / filename)
            print(f"   ✅ {filename:40s} - {description}")
        else:
            print(f"   ⚠️  {filename:40s} - NOT FOUND")
    print()

    # 2. TRAINED MODELS
    print("🤖 Organizing trained models...")
    models_dir = paper_path / "2_TRAINED_MODELS"
    models_dir.mkdir(exist_ok=True)

    any_model_found = False
    for seed in [41, 42, 43]:
        seed_dir = training_path / f"seed_{seed}"
        model_file = seed_dir / "best_model.pt"
        if model_file.exists():
            seed_models_dir = models_dir / f"seed_{seed}"
            seed_models_dir.mkdir(exist_ok=True)
            shutil.copy2(model_file, seed_models_dir / "best_model.pt")
            print(f"   ✅ Seed {seed} - best_model.pt ({model_file.stat().st_size / 1e6:.1f} MB)")
            any_model_found = True
        else:
            print(f"   ⚠️  Seed {seed} - best_model.pt NOT FOUND")

    if not any_model_found:
        print("   ⚠️  No trained model files were found.")
    print()

    # 3. METRICS DATA
    print("📈 Organizing metrics data...")
    metrics_dir = paper_path / "3_METRICS_DATA"
    metrics_dir.mkdir(exist_ok=True)

    any_metrics_found = False
    for seed in [41, 42, 43]:
        metrics_file = training_path / f"seed_{seed}" / "metrics.json"
        if metrics_file.exists():
            shutil.copy2(metrics_file, metrics_dir / f"seed_{seed}_metrics.json")
            print(f"   ✅ Seed {seed} metrics extracted")
            any_metrics_found = True
        else:
            print(f"   ⚠️  Seed {seed} metrics NOT FOUND")

    agg_metrics = training_path / "summary.json"
    if agg_metrics.exists():
        shutil.copy2(agg_metrics, metrics_dir / "aggregated_metrics.json")
        print("   ✅ Aggregated metrics (all seeds)")
        any_metrics_found = True
    else:
        print("   ⚠️  Aggregated summary.json NOT FOUND")

    if not any_metrics_found:
        print("   ⚠️  No metrics files were found.")
    print()

    # 4. TRAINING LOGS
    print("📝 Organizing training logs...")
    logs_dir = paper_path / "4_TRAINING_LOGS"
    logs_dir.mkdir(exist_ok=True)

    for log_name in ["training.log", "run_config.json"]:
        src = training_path / log_name
        if src.exists():
            shutil.copy2(src, logs_dir / log_name)
            print(f"   ✅ {log_name}")
        else:
            print(f"   ⚠️  {log_name} NOT FOUND")
    print()

    # 5. PAPER TEMPLATE
    print("📄 Creating BMVC paper templates...")
    template_dir = paper_path / "5_PAPER_TEMPLATE"
    template_dir.mkdir(exist_ok=True)
    _create_paper_templates(template_dir, training_path)

    # 6. VISUAL GUIDES
    print("🎨 Creating ML Visual Guides...")
    visuals_dir = paper_path / "6_VISUAL_GUIDES"
    visuals_dir.mkdir(exist_ok=True)
    _create_visual_guides(visuals_dir)
    print("   ✅ Architecture and Graph interpretation guides created.")

    print()
    _create_paper_writing_readme(paper_path)

    print("\n✅ RESEARCH PAPER FOLDER READY!")
    print(f"📁 Location: {paper_path}\n")


def _create_paper_templates(template_dir, training_path):
    """Create paper writing templates."""
    summary_file = training_path / "summary.json"
    summary = {}

    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

    # Abstract template
    abstract = """# ABSTRACT TEMPLATE

**Title**: Advanced Multimodal Emotion and Intention Recognition using DINOv2 and RoBERTa

**Abstract**:
This paper presents a novel multimodal approach for simultaneous emotion recognition,
intention detection, and action prediction. We propose an Advanced BEAR architecture that
replaces traditional CNNs with Meta's DINOv2 for dense visual feature extraction, fused with
RoBERTa-Large for semantic context. To overcome the extreme class imbalance inherent in
human behavioral datasets, we introduce a Dynamic Inverse-Weighted Multi-Label Focal Loss engine.
Trained on a curated "Silver Standard" dataset distilled via Large Vision Models, our system
demonstrates that state-of-the-art foundation model fusion improves emotion recognition accuracy
to {:.4f}, intention detection Macro F1-score to {:.4f}, and action prediction Macro F1-score
to {:.4f}.

**Keywords**: Multimodal Learning, DINOv2, Emotion Recognition, Multi-Label Focal Loss

---

## Key Metrics to Update:
- Emotion accuracy: {:.4f}
- Intention Macro F1: {:.4f}
- Action Macro F1: {:.4f}
""".format(
        summary.get("test_emotion_accuracy_mean", 0.0),
        summary.get("test_intention_f1_mean", 0.0),
        summary.get("test_action_f1_mean", 0.0),
        summary.get("test_emotion_accuracy_mean", 0.0),
        summary.get("test_intention_f1_mean", 0.0),
        summary.get("test_action_f1_mean", 0.0),
    )

    with open(template_dir / "ABSTRACT_TEMPLATE.md", "w", encoding="utf-8") as f:
        f.write(abstract)
    print("   ✅ Abstract template")

    # Introduction template
    intro = """# INTRODUCTION

## Motivation
Understanding human emotions and intentions from multimodal signals is crucial for
human-computer interaction, mental health monitoring, and social computing applications.

## Contributions
1. Integration of Meta's DINOv2 and RoBERTa-Large for dense semantic extraction.
2. Dual-layer attention fusion mechanism for optimal multimodal integration.
3. A novel Dynamic Inverse-Weighted Focal Loss to counter extreme long-tail class imbalances.
4. Reproducible research with multi-seed experiments (N=3) on a distilled Silver Standard dataset.

## Paper Structure
- Section 2: Related Work
- Section 3: Methodology
- Section 4: Experimental Setup
- Section 5: Results
- Section 6: Discussion
- Section 7: Conclusion
"""
    with open(template_dir / "INTRODUCTION_TEMPLATE.md", "w", encoding="utf-8") as f:
        f.write(intro)
    print("   ✅ Introduction template")

    # Methods template
    methods = """# METHODOLOGY

## 3.1 Problem Formulation

We predict:
- Emotion class: Single-label (9 classes)
- Intention labels: Multi-label (12 classes)
- Action labels: Multi-label (15 classes)

## 3.2 Advanced Architecture

### 3.2.1 Vision Foundation Backbone (DINOv2)
Unlike previous works relying on ResNet50, we utilize Meta's DINOv2 (ViT-B/14) to extract
dense, patch-level semantic features. To balance computational efficiency and domain adaptation,
the base transformer blocks are frozen, while the final two blocks remain unfrozen to adapt
to micro-expressions.

### 3.2.2 Text Foundation Backbone (RoBERTa-Large)
We employ a 355M parameter RoBERTa-Large model. The bottom 16 layers are frozen, fine-tuning
only the high-level semantic attention heads for language alignment.

### 3.2.3 Dual-Layer Attention Fusion
- Layer 1: Cross-attention between modalities for low-level feature combination.
- Layer 2: High-level semantic alignment.
- Reliability Gating: Sigmoid-based confidence scoring dynamically weights image vs. text.

## 3.3 Loss Function: Silver Standard Engine

To handle severe class imbalances, we employ a custom Weighted Multi-Task Loss:

L_total = (1.0 × L_emotion) + (2.0 × L_intention) + (2.0 × L_action)

Where L_emotion utilizes label-smoothed CrossEntropy, and the intention/action heads utilize
BCEWithLogitsLoss augmented with dynamic inverse positive-weighting (capped at 50x) to strictly
penalize minority class misclassification.
"""
    with open(template_dir / "METHODS_TEMPLATE.md", "w", encoding="utf-8") as f:
        f.write(methods)
    print("   ✅ Methods template")

    # Results template
    emotion_mean = summary.get("test_emotion_accuracy_mean", 0.0)
    emotion_std = summary.get("test_emotion_accuracy_std", 0.0)
    intention_mean = summary.get("test_intention_f1_mean", 0.0)
    intention_std = summary.get("test_intention_f1_std", 0.0)
    action_mean = summary.get("test_action_f1_mean", 0.0)
    action_std = summary.get("test_action_f1_std", 0.0)

    # 95% CI using n=3 seeds => sqrt(3) ≈ 1.732
    n_sqrt = 1.732

    results = """# RESULTS

## 4.1 Main Results (Averaged over 3 Random Seeds)

| Task | Metric | Mean | Std | 95% CI |
|------|--------|------|-----|--------|
| Emotion | Accuracy | {:.4f} | {:.4f} | [{:.4f}, {:.4f}] |
| Intention | Macro F1 | {:.4f} | {:.4f} | [{:.4f}, {:.4f}] |
| Action | Macro F1 | {:.4f} | {:.4f} | [{:.4f}, {:.4f}] |

*Note: We report Macro F1 instead of Micro F1 to rigorously demonstrate the model's ability to recognize rare classes across the long-tail distribution.*

## 4.2 Multimodal Gain
Comparison vs Single-Modality Baselines:
- Emotion: Visual-heavy reliance augmented by text context.
- Intention: Text-heavy reliance augmented by visual cues.

## 4.3 Error Analysis
The model demonstrates high robustness on the validation set, though extreme minority classes
(e.g., specific rare actions) remain challenging despite 50x focal weighting.
""".format(
        emotion_mean,
        emotion_std,
        emotion_mean - 1.96 * emotion_std / n_sqrt,
        emotion_mean + 1.96 * emotion_std / n_sqrt,
        intention_mean,
        intention_std,
        intention_mean - 1.96 * intention_std / n_sqrt,
        intention_mean + 1.96 * intention_std / n_sqrt,
        action_mean,
        action_std,
        action_mean - 1.96 * action_std / n_sqrt,
        action_mean + 1.96 * action_std / n_sqrt,
    )

    with open(template_dir / "RESULTS_TEMPLATE.md", "w", encoding="utf-8") as f:
        f.write(results)
    print("   ✅ Results template")

    # Conclusion template
    conclusion = """# CONCLUSION

This work presents Advanced BEAR, a multimodal system combining DINOv2 and RoBERTa-Large with
attention-based fusion for joint emotion, intention, and action prediction. Extensive
experiments demonstrate the efficacy of Dynamic Inverse-Weighted Focal Loss in conquering
long-tail human behavior datasets.

## References
[1] Author et al. "Paper Title" Conference Year

## Appendix
### C. Code Availability
Code available at: [GitHub URL]
Models available at: [HuggingFace URL]
"""
    with open(template_dir / "CONCLUSION_TEMPLATE.md", "w", encoding="utf-8") as f:
        f.write(conclusion)
    print("   ✅ Conclusion template")


def _create_visual_guides(visuals_dir):
    """Creates ASCII diagrams and graph interpretations for the user."""
    guide = """# ML VISUAL MURAL & INTERPRETATION GUIDE
Use these concepts to draw your diagrams for the BMVC paper and interpret your PDF graphs.

## 1. NEURAL NETWORK ARCHITECTURE FLOW
Draw this in Figma or Visio for your Methodology Section.

[IMAGE]                 [TEXT]
   │                      │
   ▼                      ▼
(DINOv2 ViT-B/14)      (RoBERTa-Large)
Frozen Base            Frozen Base (16 Layers)
Unfrozen Top           Unfrozen Top
   │                      │
   ▼                      ▼
[768-dim Vector]       [1024-dim Vector]
   │                      │
   └──► [DUAL-LAYER ATTENTION] ◄──┘
            (Fusion Block)
                  │
                  ▼
         [MASTER CONTEXT VECTOR]
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
  [EMOTION]  [INTENTION]   [ACTION]
 (9 classes) (12 classes) (15 classes)

## 2. HOW TO READ YOUR LEARNING GRAPHS
When you open `1_RESULTS_TABLES/RESEARCH_RESULTS_REPORT.pdf`, look at the Loss curve.

### SCENARIO A: "Good Fit" (What you want)
       HIGH |      •  (Train Loss)
            |       \\
            |        •
      LOSS  |         \\\\     • (Val Loss)
            |          \\\\•——/
            |           \\\\•—
       LOW  |_______________•__
              1  2  3  4  5  6
                 EPOCHS

Interpretation: The model learned the data well and generalized to the validation set.
Early stopping at Epoch 6 was the correct choice.

### SCENARIO B: "Overfit" (Memorization)
       HIGH |                   /• (Val Loss Spikes)
            |      •           /
            |       \\\\         /
      LOSS  |        •       /
            |         \\\\•——/
            |          \\\\
       LOW  |___________•______
              1  2  3  4  5  6
                 EPOCHS

Interpretation: The model started memorizing the training images instead of learning concepts.
If your PDF looks like this, it means you need to increase dropout or decrease epochs.
"""
    with open(visuals_dir / "LEARNING_CURVES_AND_ARCHITECTURE.md", "w", encoding="utf-8") as f:
        f.write(guide)


def _create_paper_writing_readme(paper_path):
    """Create README for paper writing."""
    readme = """# 📝 BMVC 2026 Research Paper Writing Guide

## Welcome! 👋
Your training has completed successfully. This folder contains the auto-generated templates reflecting your DINOv2 and RoBERTa architecture.

### 🎯 Quick Start
1. Open `1_RESULTS_TABLES/RESEARCH_RESULTS_REPORT.pdf` to view your final accuracy charts.
2. Open `6_VISUAL_GUIDES/LEARNING_CURVES_AND_ARCHITECTURE.md` to learn how to read your graphs.
3. Open `5_PAPER_TEMPLATE/ABSTRACT_TEMPLATE.md` to begin drafting your submission.

---

## 📋 Paper Sections (Use Templates!)
- **Abstract**: `5_PAPER_TEMPLATE/ABSTRACT_TEMPLATE.md`
- **Introduction**: `5_PAPER_TEMPLATE/INTRODUCTION_TEMPLATE.md`
- **Methodology**: `5_PAPER_TEMPLATE/METHODS_TEMPLATE.md`
- **Results**: `5_PAPER_TEMPLATE/RESULTS_TEMPLATE.md`
- **Conclusion**: `5_PAPER_TEMPLATE/CONCLUSION_TEMPLATE.md`

---

## 📈 The Tables You Need
**Location**: `1_RESULTS_TABLES/RESULTS_TABLE.csv`
**Also available**: `1_RESULTS_TABLES/RESULTS_LATEX_TABLE.txt` (for LaTeX)

---

## 🎓 Citation Information
If you want to cite this work:

```bibtex
@inproceedings{bmvc2026,
  title={Advanced Multimodal Emotion and Intention Recognition using DINOv2 and RoBERTa},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2026}
}
Good luck with your submission! 🎓
"""
with open(paper_path / "README_FOR_PAPER_WRITING.md", "w", encoding="utf-8") as f:
f.write(readme)

if name == "main":
output_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/professor-run"
paper_dir = sys.argv[2] if len(sys.argv) > 2 else "research_paper_data"

if Path(output_dir).exists():
    create_research_paper_folder(output_dir, paper_dir)
else:
    print(f"❌ Training output not found: {output_dir}")
    print("   Run training first: make professor-run")
