#!/usr/bin/env python3

import torch
from pathlib import Path

def find_the_truth():
    print("🔍 Opening the model's brain to count the emotion neurons...")
    
    # Path to your actual trained weights
    model_path = Path("checkpoints/professor-run/seed_41/best_model.pt")
    
    if not model_path.exists():
        print(f"❌ ERROR: Cannot find {model_path}.")
        print("This means your new training run hasn't finished its first epoch yet!")
        return

    # Load the brain
    weights = torch.load(model_path, map_location="cpu")
    
    # Search the brain for the final emotion output layer
    for key, tensor in weights.items():
        if "task_heads.emotion_head" in key and "weight" in key and tensor.dim() == 2:
            num_emotions = tensor.size(0)
            
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
            
            # Print the exact shape for proof
            print(f"Mathematical Proof (Tensor Shape): {list(tensor.shape)}")
            break

if __name__ == "__main__":
    find_the_truth()
