#!/usr/bin/env python3

import sys
import torch
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.advanced_multimodal_bear import AdvancedBEARModel

def generate_architecture_diagram():
    print("🚀 Booting up the PyTorch Engine...")
    device = torch.device("cpu") # Keep it on CPU for easy exporting
    
    print("🧠 Loading the 9-Emotion BEAR Model...")
    model = AdvancedBEARModel(hidden_dim=1024).to(device)
    model.eval()

    print("📦 Creating dummy data to trace the network paths...")
    # To map the architecture, PyTorch needs a "fake" piece of data to push through the pipes
    dummy_input_ids = torch.randint(0, 1000, (1, 128))
    dummy_attention_mask = torch.ones(1, 128)
    dummy_images = torch.randn(1, 3, 224, 224) 

    output_file = project_root / "bear_architecture.onnx"
    
    print("🗺️ Tracing the architecture and exporting...")
    with torch.no_grad():
        torch.onnx.export(
            model, 
            (dummy_input_ids, dummy_attention_mask, dummy_images), 
            str(output_file),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['Text_Input', 'Attention_Mask', 'Image_Input'],
            output_names=['Emotion_Logits', 'Intention_Logits', 'Action_Logits']
        )

    print(f"\n✅ Success! The raw architecture has been saved to: {output_file}")
    print("🎨 Next Step: Go to https://netron.app and open this file to see the diagram!")

if __name__ == "__main__":
    generate_architecture_diagram()
