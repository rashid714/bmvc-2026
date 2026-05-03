#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. SMART PATH FINDER: Guarantees it stays inside bmvc-2026
current_dir = Path.cwd()
if current_dir.name == "scripts":
    project_root = current_dir.parent
else:
    project_root = current_dir

output_dir = project_root / "research_paper_data" / "6_VISUAL_GUIDES"
output_dir.mkdir(parents=True, exist_ok=True)

# Set a clean, professional style for the academic paper
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# Exact data from your Professor Run (Seed 42)
epochs = [1, 2, 3, 4, 5, 6]
train_loss = [6.26, 4.74, 3.69, 3.11, 2.70, 2.41]
val_loss = [5.24, 3.67, 2.85, 2.34, 2.03, 1.89]

# Create the figure
plt.figure(figsize=(8, 5))

# Plot the lines
plt.plot(epochs, train_loss, marker='o', markersize=8, linewidth=2.5, label='Training Loss', color='#2c3e50')
plt.plot(epochs, val_loss, marker='s', markersize=8, linewidth=2.5, label='Validation Loss', color='#e74c3c')

# Add labels, title, and legend
plt.xlabel('Epoch', fontweight='bold', fontsize=14)
plt.ylabel('Loss', fontweight='bold', fontsize=14)
plt.title('Training vs. Validation Loss (Advanced BEAR - Seed 42)', fontweight='bold', fontsize=16)
plt.xticks(epochs)
plt.legend(frameon=True, shadow=True, fontsize=12)

# Save as both a high-res PNG and a vector PDF for LaTeX/Overleaf
plt.tight_layout()

png_path = output_dir / 'seed42_loss_curve.png'
pdf_path = output_dir / 'seed42_loss_curve.pdf'

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

print(f"✅ Successfully generated '{png_path.name}' and '{pdf_path.name}'")
print(f"📁 Saved to: {output_dir}")
