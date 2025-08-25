#!/usr/bin/env python3
"""
Resume training from the latest checkpoint.
Usage:
    python resume_training.py [checkpoint_path]
    
If no checkpoint_path is provided, it will auto-resume from the best checkpoint.
"""

import sys
import os
import subprocess

def find_latest_checkpoint():
    """Find the latest checkpoint directory"""
    checkpoints = []
    
    # Look for step checkpoints
    for item in os.listdir('.'):
        if item.startswith('checkpoint_step_') and os.path.isdir(item):
            try:
                step_num = int(item.split('checkpoint_step_')[1])
                checkpoints.append((step_num, item))
            except ValueError:
                continue
    
    if checkpoints:
        # Return the checkpoint with the highest step number
        latest_step, latest_dir = max(checkpoints)
        return latest_dir, latest_step
    
    # Fall back to best checkpoint
    if os.path.exists('checkpoint_best'):
        return 'checkpoint_best', 'best'
    
    # Fall back to final checkpoint
    if os.path.exists('checkpoint_final'):
        return 'checkpoint_final', 'final'
    
    return None, None

def main():
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint directory '{checkpoint_path}' not found!")
            return 1
        print(f"🔄 Resuming from specified checkpoint: {checkpoint_path}")
    else:
        checkpoint_path, step_info = find_latest_checkpoint()
        if checkpoint_path is None:
            print("❌ No checkpoints found!")
            print("Available options:")
            print("  - checkpoint_best (saved during training)")
            print("  - checkpoint_final (saved at end)")
            print("  - checkpoint_step_N (saved every eval_every steps)")
            return 1
        print(f"🔄 Auto-resuming from latest checkpoint: {checkpoint_path} (step {step_info})")
    
    # Modify the training script to use this checkpoint
    # We'll do this by setting an environment variable
    os.environ['RESUME_FROM_CHECKPOINT'] = checkpoint_path
    
    # Run the training script
    print(f"🚀 Starting training...")
    subprocess.run([sys.executable, 'train_distributed_llm.py'])

if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        sys.exit(exit_code)