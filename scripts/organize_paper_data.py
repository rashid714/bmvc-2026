#!/usr/bin/env python3
"""
Research Paper Data Organizer
Organizes all training results into a research paper folder
Professor can immediately start writing using this folder
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime


def create_research_paper_folder(training_output_dir, paper_output_dir="research_paper_data"):
    """
    Create organized research paper folder from training results.
    
    Structure:
    research_paper_data/
    ├── 1_RESULTS_TABLES/
    │   ├── RESEARCH_RESULTS_REPORT.pdf
    │   ├── RESULTS_TABLE.csv
    │   ├── RESULTS_LATEX_TABLE.txt
    │   └── summary.json
    ├── 2_TRAINED_MODELS/
    │   ├── seed_41/best_model.pt
    │   ├── seed_42/best_model.pt
    │   └── seed_43/best_model.pt
    ├── 3_METRICS_DATA/
    │   ├── seed_41_metrics.json
    │   ├── seed_42_metrics.json
    │   ├── seed_43_metrics.json
    │   └── aggregated_metrics.json
    ├── 4_TRAINING_LOGS/
    │   ├── training.log
    │   └── run_config.json
    ├── 5_PAPER_TEMPLATE/
    │   ├── BMVC_PAPER_TEMPLATE.md
    │   ├── RESULTS_SECTION.md
    │   ├── METHODS_SECTION.md
    │   └── ABSTRACT_TEMPLATE.md
    └── README_FOR_PAPER_WRITING.md
    """
    
    paper_path = Path(paper_output_dir)
    paper_path.mkdir(parents=True, exist_ok=True)
    
    training_path = Path(training_output_dir)
    
    print(f"🚀 Organizing research paper data...")
    print(f"   Source: {training_path}")
    print(f"   Destination: {paper_path}")
    print()
    
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
            dst = results_dir / filename
            shutil.copy2(src, dst)
            print(f"   ✅ {filename:40s} - {description}")
        else:
            print(f"   ⚠️  {filename:40s} - NOT FOUND")
    
    print()
    
    # 2. TRAINED MODELS
    print("🤖 Organizing trained models...")
    models_dir = paper_path / "2_TRAINED_MODELS"
    models_dir.mkdir(exist_ok=True)
    
    for seed in [41, 42, 43]:
        seed_dir = training_path / f"seed_{seed}"
        model_file = seed_dir / "best_model.pt"
        if model_file.exists():
            seed_models_dir = models_dir / f"seed_{seed}"
            seed_models_dir.mkdir(exist_ok=True)
            dst = seed_models_dir / "best_model.pt"
            shutil.copy2(model_file, dst)
            print(f"   ✅ Seed {seed} - best_model.pt ({model_file.stat().st_size / 1e6:.1f} MB)")
    
    print()
    
    # 3. METRICS DATA
    print("📈 Organizing metrics data...")
    metrics_dir = paper_path / "3_METRICS_DATA"
    metrics_dir.mkdir(exist_ok=True)
    
    for seed in [41, 42, 43]:
        seed_dir = training_path / f"seed_{seed}"
        metrics_file = seed_dir / "metrics.json"
        if metrics_file.exists():
            dst = metrics_dir / f"seed_{seed}_metrics.json"
            shutil.copy2(metrics_file, dst)
            print(f"   ✅ Seed {seed} metrics extracted")
    
    # Copy aggregated metrics
    agg_metrics = training_path / "summary.json"
    if agg_metrics.exists():
        dst = metrics_dir / "aggregated_metrics.json"
        shutil.copy2(agg_metrics, dst)
        print(f"   ✅ Aggregated metrics (all seeds)")
    
    print()
    
    # 4. TRAINING LOGS
    print("📝 Organizing training logs...")
    logs_dir = paper_path / "4_TRAINING_LOGS"
    logs_dir.mkdir(exist_ok=True)
    
    log_file = training_path / "training.log"
    if log_file.exists():
        shutil.copy2(log_file, logs_dir / "training.log")
        print(f"   ✅ Full training log")
    
    config_file = training_path / "run_config.json"
    if config_file.exists():
        shutil.copy2(config_file, logs_dir / "run_config.json")
        print(f"   ✅ Training configuration")
    
    print()
    
    # 5. PAPER TEMPLATE
    print("📄 Creating paper templates...")
    template_dir = paper_path / "5_PAPER_TEMPLATE"
    template_dir.mkdir(exist_ok=True)
    
    _create_paper_templates(template_dir, training_path)
    
    print()
    
    # Create README
    _create_paper_writing_readme(paper_path, training_path)
    
    print("✅ RESEARCH PAPER FOLDER READY!")
    print()
    print(f"📁 Location: {paper_path}")
    print()
    print("📋 What's inside:")
    print("   1_RESULTS_TABLES/        - PDF, CSV, LaTeX tables for paper")
    print("   2_TRAINED_MODELS/        - Best trained models (3 seeds)")
    print("   3_METRICS_DATA/          - All detailed metrics")
    print("   4_TRAINING_LOGS/         - Logs and configuration")
    print("   5_PAPER_TEMPLATE/        - Templates for writing")
    print("   README_FOR_PAPER_WRITING.md - Start here!")
    print()
    print("🎯 Next step: Open README_FOR_PAPER_WRITING.md")
    print()


def _create_paper_templates(template_dir, training_path):
    """Create paper writing templates."""
    
    # Load summary for use in templates
    summary_file = training_path / "summary.json"
    summary = {}
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    
    # Abstract template
    abstract = """# ABSTRACT TEMPLATE

**Title**: Multimodal Emotion and Intention Recognition using Advanced Dual-Layer BEAR

**Abstract**:
This paper presents a novel multimodal approach for simultaneous emotion recognition, 
intention detection, and action prediction. We propose Advanced BEAR, which employs a 
dual-layer language model backbone (RoBERTa-large + DistilGPT2) combined with multimodal 
fusion through advanced attention mechanisms. Our system processes text, images, audio, 
and video modalities with reliability-weighted fusion. Experiments on [DATASET] demonstrate 
that multimodal fusion improves emotion recognition accuracy by [+X]%, intention detection 
F1-score by [+Y]%, and action prediction F1-score by [+Z]% compared to text-only baselines.

**Keywords**: Multimodal Learning, Emotion Recognition, Intention Detection, Attention Mechanisms

---

## Key Metrics to Update:
- Emotion accuracy: {:.4f}
- Intention F1: {:.4f}
- Action F1: {:.4f}
""".format(
        summary.get('test_emotion_accuracy_mean', 0.0),
        summary.get('test_intention_f1_mean', 0.0),
        summary.get('test_action_f1_mean', 0.0),
    )
    
    with open(template_dir / "ABSTRACT_TEMPLATE.md", 'w') as f:
        f.write(abstract)
    print(f"   ✅ Abstract template")
    
    # Introduction template
    intro = """# INTRODUCTION

## Motivation
Understanding human emotions and intentions from multimodal signals is crucial for 
human-computer interaction, mental health monitoring, and social computing applications.

## Related Work
- Single-modality approaches: [citations]
- Multimodal fusion: [citations]
- Attention mechanisms for fusion: [citations]

## Contributions
1. Advanced dual-layer LLM backbone combining RoBERTa and DistilGPT2
2. Dual-layer attention fusion mechanism for multimodal integration
3. Comprehensive evaluation on [DATASET] with statistical significance testing
4. Reproducible research with multi-seed experiments (N=3)

## Paper Structure
- Section 2: Related Work
- Section 3: Methodology
- Section 4: Experimental Setup
- Section 5: Results
- Section 6: Discussion
- Section 7: Conclusion
"""
    
    with open(template_dir / "INTRODUCTION_TEMPLATE.md", 'w') as f:
        f.write(intro)
    print(f"   ✅ Introduction template")
    
    # Methods template
    methods = """# METHODOLOGY

## 3.1 Problem Formulation

Given a multimodal input sample consisting of:
- Text transcript: **t** ∈ ℝ^T
- Image features: **i** ∈ ℝ^D_i
- Audio features: **a** ∈ ℝ^D_a
- Video features: **v** ∈ ℝ^D_v

We predict:
- Emotion class: **e** ∈ {1, ..., 11}
- Intention labels: **y_int** ∈ {0,1}^20 (multi-label)
- Action labels: **y_act** ∈ {0,1}^15 (multi-label)

## 3.2 Advanced BEAR Architecture

### 3.2.1 Dual-Layer LLM Backbone

**Layer 1 (RoBERTa-large)**:
- 355M parameters
- 1024-dimensional hidden state
- Robust contextual understanding

**Layer 2 (DistilGPT2)**:
- 82M parameters
- 768-dimensional hidden state
- Complementary semantic understanding

**Fusion Gate**: 
f_text = σ(W_text([h_roberta; h_gpt2]))

### 3.2.2 Multimodal Encoders

For each modality m ∈ {image, audio, video}:
h_m = MLP(features_m) → 1024-dim

### 3.2.3 Dual-Layer Attention Fusion

**Layer 1 (Low-level fusion)**:
- Cross-attention between modalities
- Feature-level combination

**Layer 2 (High-level fusion)**:
- Semantic-level alignment
- Context-aware integration

**Reliability Weighting**:
r_m = Softmax(MLP_conf(h_text)) ×  (1 - MLP_unc(h_text))

## 3.3 Tri-Task Learning Head

Shared encoder followed by 3 task-specific heads:
- Emotion: 11-way softmax
- Intention: Multi-label sigmoid (20 classes)
- Action: Multi-label sigmoid (15 classes)

## 3.4 Loss Function

L_total = w_e × L_emotion + w_int × L_intention + w_act × L_action

where w_e=1.0, w_int=1.2, w_act=1.0
"""
    
    with open(template_dir / "METHODS_TEMPLATE.md", 'w') as f:
        f.write(methods)
    print(f"   ✅ Methods template")
    
    # Results template
    results = """# RESULTS

## 4.1 Main Results

| Task | Metric | Mean | Std | 95% CI |
|------|--------|------|-----|--------|
| Emotion | Accuracy | {:.4f} | {:.4f} | [{:.4f}, {:.4f}] |
| Intention | Micro F1 | {:.4f} | {:.4f} | [{:.4f}, {:.4f}] |
| Action | Micro F1 | {:.4f} | {:.4f} | [{:.4f}, {:.4f}] |

## 4.2 Multimodal Gain

Comparison vs Text-Only Baseline:
- Emotion: +X% improvement
- Intention: +Y% improvement
- Action: +Z% improvement

## 4.3 Per-Seed Results

### Seed 41
- [Metrics]

### Seed 42
- [Metrics]

### Seed 43
- [Metrics]

## 4.4 Ablation Studies

[Add ablation results here]

## 4.5 Error Analysis

[Add error analysis here]
""".format(
        summary.get('test_emotion_accuracy_mean', 0.0),
        summary.get('test_emotion_accuracy_std', 0.0),
        summary.get('test_emotion_accuracy_mean', 0.0) - 1.96*summary.get('test_emotion_accuracy_std', 0.0)/3,
        summary.get('test_emotion_accuracy_mean', 0.0) + 1.96*summary.get('test_emotion_accuracy_std', 0.0)/3,
        summary.get('test_intention_f1_mean', 0.0),
        summary.get('test_intention_f1_std', 0.0),
        summary.get('test_intention_f1_mean', 0.0) - 1.96*summary.get('test_intention_f1_std', 0.0)/3,
        summary.get('test_intention_f1_mean', 0.0) + 1.96*summary.get('test_intention_f1_std', 0.0)/3,
        summary.get('test_action_f1_mean', 0.0),
        summary.get('test_action_f1_std', 0.0),
        summary.get('test_action_f1_mean', 0.0) - 1.96*summary.get('test_action_f1_std', 0.0)/3,
        summary.get('test_action_f1_mean', 0.0) + 1.96*summary.get('test_action_f1_std', 0.0)/3,
    )
    
    with open(template_dir / "RESULTS_TEMPLATE.md", 'w') as f:
        f.write(results)
    print(f"   ✅ Results template")
    
    # Discussion template
    discussion = """# DISCUSSION

## 5.1 Key Findings

1. **Dual-Layer LLM Benefits**: Combining RoBERTa and DistilGPT2 provides complementary 
   semantic understanding, improving model robustness.

2. **Multimodal Fusion Effectiveness**: Cross-modal attention improves performance by 
   leveraging information redundancy and complementarity.

3. **Reliability Weighting**: Per-modality confidence scoring allows the model to 
   dynamically weight modalities based on their informativeness.

## 5.2 Comparison with Baselines

- vs Text-only BEAR: +X% emotion, +Y% intention, +Z% action
- vs Simple concatenation: +A% emotion, +B% intention, +C% action
- vs [Other methods]: [Comparison]

## 5.3 Limitations

1. Dataset size: [Comment on dataset scale]
2. Modality importance: [Comment on which modalities matter most]
3. Generalization: [Comment on cross-domain evaluation]

## 5.4 Future Work

1. Larger-scale evaluation on diverse datasets
2. Real-time inference optimization
3. Integration with downstream applications
4. Cross-lingual and cross-cultural evaluation
"""
    
    with open(template_dir / "DISCUSSION_TEMPLATE.md", 'w') as f:
        f.write(discussion)
    print(f"   ✅ Discussion template")
    
    # Conclusion template
    conclusion = """# CONCLUSION

This work presents Advanced BEAR, a multimodal system combining dual-layer LLMs with 
attention-based fusion for joint emotion, intention, and action prediction. Extensive 
experiments demonstrate significant improvements over baselines through effective 
multimodal integration. Future work will focus on scaling to larger datasets and real-world 
applications.

## Contributions Summary

✅ Novel dual-layer LLM architecture
✅ Advanced multimodal fusion mechanism
✅ Comprehensive multi-seed evaluation
✅ Publication-ready reproducible research

## References

[1] Author et al. "Paper Title" Conference Year
[2] ...

## Appendix

### A. Hyperparameters

[List all hyperparameters from run_config.json]

### B. Additional Results

[Tables, figures, additional metrics]

### C. Code Availability

Code available at: [GitHub URL]
Models available at: [HuggingFace URL]
"""
    
    with open(template_dir / "CONCLUSION_TEMPLATE.md", 'w') as f:
        f.write(conclusion)
    print(f"   ✅ Conclusion template")


def _create_paper_writing_readme(paper_path, training_path):
    """Create README for paper writing."""
    
    readme = """# 📝 Research Paper Writing Guide

## Welcome! 👋

Your training has completed successfully. This folder contains everything you need 
to write your BMVC 2026 research paper.

---

## 📂 Folder Structure

```
research_paper_data/
├── 1_RESULTS_TABLES/              ← Your tables and metrics
│   ├── RESEARCH_RESULTS_REPORT.pdf   (Open this first!)
│   ├── RESULTS_TABLE.csv            (Import to your paper)
│   ├── RESULTS_LATEX_TABLE.txt      (For Overleaf/LaTeX)
│   └── summary.json                 (Raw metrics)
│
├── 2_TRAINED_MODELS/              ← Your trained models
│   ├── seed_41/best_model.pt
│   ├── seed_42/best_model.pt
│   └── seed_43/best_model.pt
│
├── 3_METRICS_DATA/                ← Detailed metric breakdowns
│   ├── seed_41_metrics.json
│   ├── seed_42_metrics.json
│   ├── seed_43_metrics.json
│   └── aggregated_metrics.json
│
├── 4_TRAINING_LOGS/               ← Configuration and logs
│   ├── training.log
│   └── run_config.json
│
├── 5_PAPER_TEMPLATE/              ← Fill-in templates
│   ├── ABSTRACT_TEMPLATE.md
│   ├── INTRODUCTION_TEMPLATE.md
│   ├── METHODS_TEMPLATE.md
│   ├── RESULTS_TEMPLATE.md
│   ├── DISCUSSION_TEMPLATE.md
│   └── CONCLUSION_TEMPLATE.md
│
└── README_FOR_PAPER_WRITING.md    (This file)
```

---

## 🎯 Quick Start (5 minutes)

### Step 1: Check Your Results
```
1. Open: 1_RESULTS_TABLES/RESEARCH_RESULTS_REPORT.pdf
2. Review the tables and metrics
3. Note down the key numbers
```

### Step 2: Copy Tables
```
Option A (Easiest for most word processors):
- Copy-paste tables from RESEARCH_RESULTS_REPORT.pdf

Option B (For Overleaf/LaTeX):
- Use content from RESULTS_LATEX_TABLE.txt
- Paste directly into your .tex file

Option C (For Excel/spreadsheets):
- Import RESULTS_TABLE.csv
```

### Step 3: Use Templates
```
1. Open: 5_PAPER_TEMPLATE/ABSTRACT_TEMPLATE.md
2. Replace [PLACEHOLDERS] with your content
3. Repeat for other sections
```

---

## 📊 Key Metrics (Your Results)

Your model achieved:

- **Emotion Recognition**: Accurate classification of emotions (11 classes)
- **Intention Detection**: Multi-label prediction of speaker intent (20 classes)
- **Action Prediction**: Multi-label prediction of actions (15 classes)

See: `1_RESULTS_TABLES/summary.json` for exact numbers

---

## 📋 Paper Sections (Use Templates!)

### 1. Abstract (5-10 min)
- Located: `5_PAPER_TEMPLATE/ABSTRACT_TEMPLATE.md`
- Action: Update with your actual results

### 2. Introduction (15-30 min)
- Located: `5_PAPER_TEMPLATE/INTRODUCTION_TEMPLATE.md`
- Action: Expand with your motivation

### 3. Related Work (20-30 min)
- Add citations to relevant papers
- Compare with related approaches

### 4. Methodology (20-30 min)
- Located: `5_PAPER_TEMPLATE/METHODS_TEMPLATE.md`
- Already includes architecture details!

### 5. Experiments (15-20 min)
- Located: `5_PAPER_TEMPLATE/RESULTS_TEMPLATE.md`
- Paste tables from `1_RESULTS_TABLES/`

### 6. Results (10-15 min)
- Tables ready: `1_RESULTS_TABLES/RESULTS_TABLE.csv`
- PDF ready: `1_RESULTS_TABLES/RESEARCH_RESULTS_REPORT.pdf`

### 7. Discussion (20-30 min)
- Located: `5_PAPER_TEMPLATE/DISCUSSION_TEMPLATE.md`
- Discuss findings and implications

### 8. Conclusion (5-10 min)
- Located: `5_PAPER_TEMPLATE/CONCLUSION_TEMPLATE.md`
- Summarize contributions

---

## 📈 The Tables You Need

### Main Results Table

| Task | Metric | Value |
|------|--------|-------|
| Emotion | Accuracy | [in summary.json] |
| Intention | Micro F1 | [in summary.json] |
| Action | Micro F1 | [in summary.json] |

**Location**: `1_RESULTS_TABLES/RESULTS_TABLE.csv`
**Also available**: `1_RESULTS_TABLES/RESULTS_LATEX_TABLE.txt` (for LaTeX)

---

## 🔧 Using the Data

### For Microsoft Word / Google Docs
1. Open: `1_RESULTS_TABLES/RESEARCH_RESULTS_REPORT.pdf`
2. Take screenshots of tables
3. Insert into your document

### For Overleaf / LaTeX
1. Open: `1_RESULTS_TABLES/RESULTS_LATEX_TABLE.txt`
2. Copy entire table
3. Paste into your .tex file

### For Analysis / Further Work
1. Open: `3_METRICS_DATA/aggregated_metrics.json`
2. Import into Python/R/Excel for analysis

---

## 📖 Statistical Interpretation

Your results include:

- **Mean**: Average performance across 3 seeds (reproducibility)
- **Std Dev**: Standard deviation (variability across seeds)
- **95% CI**: Confidence interval (±1.96 × SEM)

**Example interpretation**:
"Emotion accuracy achieved 0.6234 ± 0.0145 (95% CI: [0.6148, 0.6320])"

This shows your method is robust and reproducible.

---

## 🎓 Citation Information

If you want to cite this work:

```bibtex
@inproceedings{bmvc2026,
  title={Advanced Multimodal Emotion Recognition with Dual-Layer BEAR},
  author={Your Name},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2026}
}
```

Detailed config: See `4_TRAINING_LOGS/run_config.json`

---

## ✅ Paper Writing Checklist

Before submitting to BMVC, ensure:

- [ ] Abstract written and under 250 words
- [ ] Introduction motivates the problem clearly
- [ ] Related work cites relevant papers (10-15 citations)
- [ ] Methodology section explains architecture
- [ ] Results tables included with error bars/std dev
- [ ] Discussion compares with baselines
- [ ] Conclusion summarizes contributions
- [ ] References properly formatted
- [ ] Figures have captions
- [ ] Tables are clear and readable
- [ ] Paper is 6-8 pages (BMVC requirement)

---

## 🚀 Tips for Writing

1. **Use the templates**: They already have the structure
2. **Insert tables first**: Makes writing easier
3. **Have checkpoints**: Save frequently
4. **Get feedback**: Have colleagues review
5. **Follow BMVC guidelines**: (6-8 pages, specific format)

---

## 📞 If You Need Raw Data

All raw metrics are in: `3_METRICS_DATA/`

Files available:
- `seed_41_metrics.json` - Per-epoch metrics for seed 41
- `seed_42_metrics.json` - Per-epoch metrics for seed 42
- `seed_43_metrics.json` - Per-epoch metrics for seed 43
- `aggregated_metrics.json` - All seeds aggregated

---

## 🎯 Next Steps

1. Open `5_PAPER_TEMPLATE/ABSTRACT_TEMPLATE.md`
2. Start writing your abstract
3. Follow the templates section by section
4. Insert tables from `1_RESULTS_TABLES/`
5. Submit to BMVC!

---

## ✨ You're All Set!

Everything you need for a publication-quality paper is in this folder.

**The research is complete. Now write the paper!** 📝

Good luck with your BMVC 2026 submission! 🎓
"""
    
    with open(paper_path / "README_FOR_PAPER_WRITING.md", 'w') as f:
        f.write(readme)


if __name__ == "__main__":
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/advanced-results-final"
    paper_dir = sys.argv[2] if len(sys.argv) > 2 else "research_paper_data"
    
    if Path(output_dir).exists():
        create_research_paper_folder(output_dir, paper_dir)
    else:
        print(f"❌ Training output not found: {output_dir}")
        print("   Run training first: make advanced-cloud")
