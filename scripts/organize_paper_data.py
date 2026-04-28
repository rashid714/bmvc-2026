#!/usr/bin/env python3

import json
import shutil
import sys
from pathlib import Path

def create_research_paper_folder(training_output_dir, paper_output_dir):
    paper_path = Path(paper_output_dir)
    paper_path.mkdir(parents=True, exist_ok=True)
    training_path = Path(training_output_dir)
    
    print("🚀 Organizing research paper data...")
    print(f"   Source: {training_path}")
    print(f"   Destination: {paper_path}\n")
    
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
    
    print("🤖 Organizing trained models...")
    models_dir = paper_path / "2_TRAINED_MODELS"
    models_dir.mkdir(exist_ok=True)
    
    any_model_found = False
    for seed in [1, 2, 3, 41, 42, 43]: 
        seed_dir = training_path / f"seed_{seed}"
        model_file = seed_dir / "best_model.pt"
        if model_file.exists():
            seed_models_dir = models_dir / f"seed_{seed}"
            seed_models_dir.mkdir(exist_ok=True)
            shutil.copy2(model_file, seed_models_dir / "best_model.pt")
            print(f"   ✅ Seed {seed} - best_model.pt ({model_file.stat().st_size / 1e6:.1f} MB)")
            any_model_found = True
            
    if not any_model_found:
        print("   ⚠️  No trained model files were found.")
    print()
    
    print("📈 Organizing metrics data...")
    metrics_dir = paper_path / "3_METRICS_DATA"
    metrics_dir.mkdir(exist_ok=True)
    
    any_metrics_found = False
    for seed in [1, 2, 3, 41, 42, 43]:
        metrics_file = training_path / f"seed_{seed}" / "metrics.json"
        if metrics_file.exists():
            shutil.copy2(metrics_file, metrics_dir / f"seed_{seed}_metrics.json")
            print(f"   ✅ Seed {seed} metrics extracted")
            any_metrics_found = True

    agg_metrics = training_path / "summary.json"
    if agg_metrics.exists():
        shutil.copy2(agg_metrics, metrics_dir / "aggregated_metrics.json")
        print("   ✅ Aggregated metrics (all seeds)")
        any_metrics_found = True
        
    if not any_metrics_found:
        print("   ⚠️  No metrics files were found.")
    print()
    
    print("📝 Organizing training logs...")
    logs_dir = paper_path / "4_TRAINING_LOGS"
    logs_dir.mkdir(exist_ok=True)
    
    for log_name in ["training.log", "run_config.json"]:
        src = training_path / log_name
        if src.exists():
            shutil.copy2(src, logs_dir / log_name)
            print(f"   ✅ {log_name}")
            
    print()
    print("📄 Creating BMVC paper templates...")
    template_dir = paper_path / "5_PAPER_TEMPLATE"
    template_dir.mkdir(exist_ok=True)
    _create_paper_templates(template_dir, training_path)
    
    print("🎨 Creating ML Visual Guides...")
    visuals_dir = paper_path / "6_VISUAL_GUIDES"
    visuals_dir.mkdir(exist_ok=True)
    _create_visual_guides(visuals_dir)
    print("   ✅ Architecture and Graph interpretation guides created.")
    print("\n✅ RESEARCH PAPER FOLDER READY!")
    print(f"📁 Location: {paper_path}\n")

def _create_paper_templates(template_dir, training_path):
    summary_file = training_path / "summary.json"
    summary = {}
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
            
    emotion_mean = summary.get("test_emotion_accuracy_mean", 0.0)
    emotion_std = summary.get("test_emotion_accuracy_std", 0.0)
    intention_mean = summary.get("test_intention_f1_mean", 0.0)
    intention_std = summary.get("test_intention_f1_std", 0.0)
    action_mean = summary.get("test_action_f1_mean", 0.0)
    action_std = summary.get("test_action_f1_std", 0.0)
    n_sqrt = 1.732

    abstract = f"""# ABSTRACT TEMPLATE\n\n**Title**: Advanced Multimodal Emotion and Intention Recognition using DINOv2 and RoBERTa\n\n**Abstract**:\nThis paper presents a novel multimodal approach for simultaneous emotion recognition,\nintention detection, and action prediction. We propose an Advanced BEAR architecture that\nreplaces traditional CNNs with Meta's DINOv2 for dense visual feature extraction, fused with\nRoBERTa-Large for semantic context. To overcome the extreme class imbalance inherent in\nhuman behavioral datasets, we introduce a Dynamic Inverse-Weighted Multi-Label Focal Loss engine.\nTrained on a curated "Silver Standard" dataset distilled via Large Vision Models, our system\ndemonstrates that state-of-the-art foundation model fusion improves emotion recognition accuracy\nto {emotion_mean:.4f}, intention detection Macro F1-score to {intention_mean:.4f}, and action prediction Macro F1-score\nto {action_mean:.4f}.\n\n**Keywords**: Multimodal Learning, DINOv2, Emotion Recognition, Multi-Label Focal Loss\n\n---\n\n## Key Metrics to Update:\n- Emotion accuracy: {emotion_mean:.4f}\n- Intention Macro F1: {intention_mean:.4f}\n- Action Macro F1: {action_mean:.4f}\n"""
    
    with open(template_dir / "ABSTRACT_TEMPLATE.md", "w", encoding="utf-8") as f:
        f.write(abstract)
    print("   ✅ Abstract template")

    intro = """# INTRODUCTION\n\n## Motivation\nUnderstanding human emotions and intentions from multimodal signals is crucial for\nhuman-computer interaction, mental health monitoring, and social computing applications.\n\n## Contributions\n1. Integration of Meta's DINOv2 and RoBERTa-Large for dense semantic extraction.\n2. Dual-layer attention fusion mechanism for optimal multimodal integration.\n3. A novel Dynamic Inverse-Weighted Focal Loss to counter extreme long-tail class imbalances.\n4. Reproducible research with multi-seed experiments (N=3) on a distilled Silver Standard dataset.\n\n## Paper Structure\n- Section 2: Related Work\n- Section 3: Methodology\n- Section 4: Experimental Setup\n- Section 5: Results\n- Section 6: Discussion\n- Section 7: Conclusion\n"""
    with open(template_dir / "INTRODUCTION_TEMPLATE.md", "w", encoding="utf-8") as f:
        f.write(intro)
    print("   ✅ Introduction template")

    methods = """# METHODOLOGY\n\n## 3.1 Problem Formulation\n\nWe predict:\n- Emotion class: Single-label (9 classes)\n- Intention labels: Multi-label (12 classes)\n- Action labels: Multi-label (15 classes)\n\n## 3.2 Advanced Architecture\n\n### 3.2.1 Vision Foundation Backbone (DINOv2)\nUnlike previous works relying on ResNet50, we utilize Meta's DINOv2 (ViT-B/14) to extract\ndense, patch-level semantic features. To balance computational efficiency and domain adaptation,\nthe base transformer blocks are frozen, while the final two blocks remain unfrozen to adapt\nto micro-expressions.\n\n### 3.2.2 Text Foundation Backbone (RoBERTa-Large)\nWe employ a 355M parameter RoBERTa-Large model. The bottom 16 layers are frozen, fine-tuning\nonly the high-level semantic attention heads for language alignment.\n\n### 3.2.3 Dual-Layer Attention Fusion\n- Layer 1: Cross-attention between modalities for low-level feature combination.\n- Layer 2: High-level semantic alignment.\n- Reliability Gating: Sigmoid-based confidence scoring dynamically weights image vs. text.\n\n## 3.3 Loss Function: Silver Standard Engine\n\nTo handle severe class imbalances, we employ a custom Weighted Multi-Task Loss:\n\nL_total = (1.0 × L_emotion) + (2.0 × L_intention) + (2.0 × L_action)\n\nWhere L_emotion utilizes label-smoothed CrossEntropy, and the intention/action heads utilize\nBCEWithLogitsLoss augmented with dynamic inverse positive-weighting (capped at 50x) to strictly\npenalize minority class misclassification.\n"""
    with open(template_dir / "METHODS_TEMPLATE.md", "w", encoding="utf-8") as f:
        f.write(methods)
    print("   ✅ Methods template")

    results = f"""# RESULTS\n\n## 4.1 Main Results (Averaged over 3 Random Seeds)\n\n| Task | Metric | Mean | Std | 95% CI |\n|------|--------|------|-----|--------|\n| Emotion | Accuracy | {emotion_mean:.4f} | {emotion_std:.4f} | [{emotion_mean - 1.96 * emotion_std / n_sqrt:.4f}, {emotion_mean + 1.96 * emotion_std / n_sqrt:.4f}] |\n| Intention | Macro F1 | {intention_mean:.4f} | {intention_std:.4f} | [{intention_mean - 1.96 * intention_std / n_sqrt:.4f}, {intention_mean + 1.96 * intention_std / n_sqrt:.4f}] |\n| Action | Macro F1 | {action_mean:.4f} | {action_std:.4f} | [{action_mean - 1.96 * action_std / n_sqrt:.4f}, {action_mean + 1.96 * action_std / n_sqrt:.4f}] |\n\n*Note: We report Macro F1 instead of Micro F1 to rigorously demonstrate the model's ability to recognize rare classes across the long-tail distribution.*\n\n## 4.2 Multimodal Gain\nComparison vs Single-Modality Baselines:\n- Emotion: Visual-heavy reliance augmented by text context.\n- Intention: Text-heavy reliance augmented by visual cues.\n\n## 4.3 Error Analysis\nThe model demonstrates high robustness on the validation set, though extreme minority classes\n(e.g., specific rare actions) remain challenging despite 50x focal weighting.\n"""
    with open(template_dir / "RESULTS_TEMPLATE.md", "w", encoding="utf-8") as f:
        f.write(results)
    print("   ✅ Results template")

    conclusion = """# CONCLUSION\n\nThis work presents Advanced BEAR, a multimodal system combining DINOv2 and RoBERTa-Large with\nattention-based fusion for joint emotion, intention, and action prediction. Extensive\nexperiments demonstrate the efficacy of Dynamic Inverse-Weighted Focal Loss in conquering\nlong-tail human behavior datasets.\n\n## References\n[1] Author et al. "Paper Title" Conference Year\n\n## Appendix\n### C. Code Availability\nCode available at: [GitHub URL]\nModels available at: [HuggingFace URL]\n"""
    with open(template_dir / "CONCLUSION_TEMPLATE.md", "w", encoding="utf-8") as f:
        f.write(conclusion)
    print("   ✅ Conclusion template")

def _create_visual_guides(visuals_dir):
    guide = """# ML VISUAL MURAL & INTERPRETATION GUIDE\nUse these concepts to draw your diagrams for the BMVC paper and interpret your PDF graphs.\n\n## 1. NEURAL NETWORK ARCHITECTURE FLOW\nDraw this in Figma or Visio for your Methodology Section.\n\n[IMAGE]                 [TEXT]\n   │                      │\n   ▼                      ▼\n(DINOv2 ViT-B/14)      (RoBERTa-Large)\nFrozen Base            Frozen Base (16 Layers)\nUnfrozen Top           Unfrozen Top\n   │                      │\n   ▼                      ▼\n[768-dim Vector]       [1024-dim Vector]\n   │                      │\n   └──► [DUAL-LAYER ATTENTION] ◄──┘\n            (Fusion Block)\n                  │\n                  ▼\n         [MASTER CONTEXT VECTOR]\n                  │\n      ┌───────────┼───────────┐\n      ▼           ▼           ▼\n  [EMOTION]  [INTENTION]   [ACTION]\n (9 classes) (12 classes) (15 classes)\n\n## 2. HOW TO READ YOUR LEARNING GRAPHS\nWhen you open `1_RESULTS_TABLES/RESEARCH_RESULTS_REPORT.pdf`, look at the Loss curve.\n\n### SCENARIO A: "Good Fit" (What you want)\n       HIGH |      •  (Train Loss)\n            |       \\\n            |        •\n      LOSS  |         \\\\     • (Val Loss)\n            |          \\\\•——/\n            |           \\\\•—\n       LOW  |_______________•__\n              1  2  3  4  5  6\n                 EPOCHS\n\nInterpretation: The model learned the data well and generalized to the validation set.\nEarly stopping at Epoch 6 was the correct choice.\n\n### SCENARIO B: "Overfit" (Memorization)\n       HIGH |                   /• (Val Loss Spikes)\n            |      •           /\n            |       \\\\         /\n      LOSS  |        •       /\n            |         \\\\•——/\n            |          \\\\\n       LOW  |___________•______\n              1  2  3  4  5  6\n                 EPOCHS\n\nInterpretation: The model started memorizing the training images instead of learning concepts.\nIf your PDF looks like this, it means you need to increase dropout or decrease epochs.\n"""
    with open(visuals_dir / "LEARNING_CURVES_AND_ARCHITECTURE.md", "w", encoding="utf-8") as f:
        f.write(guide)

if _name_ == "__main__":
    # Locks the path relative to the script to guarantee it saves directly inside bmvc-2026
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "checkpoints" / "professor-run"
    paper_dir = project_root / "research_paper_data"

    if output_dir.exists():
        create_research_paper_folder(output_dir, paper_dir)
    else:
        print(f"❌ Training output not found: {output_dir}")
        print("   Run training first: make professor-run")
