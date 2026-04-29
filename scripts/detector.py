#!/usr/bin/env python3

import torch
from pathlib import Path

def find_the_truth():
    print("🔍 Opening the model's brain to find the FINAL classification layer...")
    
    # Path to your actual trained weights
    model_path = Path("checkpoints/professor-run/seed_41/best_model.pt")
    
    if not model_path.exists():
        print(f"❌ ERROR: Cannot find {model_path}.")
        return

    # Load the brain (silencing the warning with weights_only=True)
    weights = torch.load(model_path, map_location="cpu", weights_only=True)
    
    # We specifically want Layer 3, which is the final output node!
    target_key = "task_heads.emotion_head.3.weight"
    
    if target_key in weights:
        tensor = weights[target_key]
        num_emotions = tensor.size(0) # This gets the final output size
        
        print("\n" + "═"*50)
        if num_emotions == 9:
            print(" 🎉 THE ABSOLUTE TRUTH: Your model has EXACTLY 9 emotions!")
            print("    (You are perfectly set up for the new clean run!)")
        elif num_emotions == 11:
            print(" ⚠️ THE ABSOLUTE TRUTH: Your model has 11 emotions.")
            print("    (This means it is still loading the OLD training run.)")
        else:
            print(f" 🤔 Wait, your model has {num_emotions} emotions? Something is weird.")
        print("═"*50 + "\n")
        
        # Print the exact shape for proof (Should be [9, 512] or [11, 512])
        print(f"Mathematical Proof (Tensor Shape): {list(tensor.shape)}")
    else:
        print(f"❌ Could not find '{target_key}' in the weights.")
        print("Available keys in task_heads.emotion_head:")
        for k in weights.keys():
            if "task_heads.emotion_head" in k:
                print(f"  - {k}")

if __name__ == "__main__":
    find_the_truth()
