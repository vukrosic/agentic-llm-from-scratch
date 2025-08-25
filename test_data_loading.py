#!/usr/bin/env python3
"""
Test script to verify the hermes_reasoning_tool_use dataset loading works correctly.
This script will load a small portion of the dataset to test the functionality.
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from accelerate import Accelerator
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class TestModelConfig:
    # Minimal config for testing
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 24
    max_steps: int = 100  # Reduced for testing
    
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01
    
    max_seq_len: int = 512
    
    eval_every: int = 50
    eval_steps: int = 10
    
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0
    
    mixed_precision: str = "fp16"
    vocab_size: Optional[int] = None

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

def extract_conversation_text(conversation):
    """Extract and concatenate text from conversation format"""
    extracted_text = ""
    for message in conversation:
        role = message.get('from', '')
        content = message.get('value', '')
        
        # Add role prefix for context
        if role == 'system':
            extracted_text += f"<|system|>\n{content}\n\n"
        elif role == 'human':
            extracted_text += f"<|user|>\n{content}\n\n"
        elif role == 'gpt':
            extracted_text += f"<|assistant|>\n{content}\n\n"
        else:
            extracted_text += f"<|{role}|>\n{content}\n\n"
    
    return extracted_text

def test_dataset_loading():
    """Test loading and processing a small portion of the dataset"""
    print("🧪 Testing hermes_reasoning_tool_use dataset loading...")
    
    # Initialize accelerator for testing
    accelerator = Accelerator(mixed_precision="no")  # No mixed precision for testing
    
    # Load tokenizer
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"   Vocabulary size: {tokenizer.vocab_size:,}")
    
    # Load a small portion of the dataset for testing
    print("📥 Loading dataset (first 100 examples for testing)...")
    dataset = load_dataset("interstellarninja/hermes_reasoning_tool_use", split="train[:100]", token=False)
    
    print(f"   Loaded {len(dataset)} conversations for testing")
    
    # Process conversations
    print("🔄 Processing conversations...")
    texts = []
    total_chars = 0
    
    for i, item in enumerate(dataset):
        conversation = item["conversations"]
        extracted_text = extract_conversation_text(conversation)
        texts.append(extracted_text)
        total_chars += len(extracted_text)
        
        # Show first conversation as example
        if i == 0:
            print(f"\n📝 Example conversation {i+1}:")
            print("=" * 50)
            print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
            print("=" * 50)
    
    print(f"📊 Processing results:")
    print(f"   Total conversations: {len(texts)}")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Average chars per conversation: {total_chars / len(texts):.1f}")
    
    # Tokenize
    print("🔤 Tokenizing conversations...")
    all_tokens = []
    
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        all_tokens.extend(tokens)
        
        if i == 0:
            print(f"   First conversation tokens: {len(tokens)}")
    
    print(f"📊 Tokenization results:")
    print(f"   Total tokens: {len(all_tokens):,}")
    print(f"   Average tokens per conversation: {len(all_tokens) / len(texts):.1f}")
    
    # Create sample output file
    sample_file = "sample_hermes_data.txt"
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write("# Sample from hermes_reasoning_tool_use dataset\n\n")
        f.write(f"Test Statistics (first 100 conversations):\n")
        f.write(f"- Total conversations: {len(texts):,}\n")
        f.write(f"- Total characters: {total_chars:,}\n")
        f.write(f"- Average chars per conversation: {total_chars / len(texts):.1f}\n")
        f.write(f"- Total tokens: {len(all_tokens):,}\n")
        f.write(f"- Average tokens per conversation: {len(all_tokens) / len(texts):.1f}\n")
        f.write(f"- Vocabulary size: {tokenizer.vocab_size:,}\n\n")
        
        # Write first 3 conversations as samples
        for i in range(min(3, len(texts))):
            f.write(f"=== Conversation {i+1} ===\n")
            f.write(texts[i])
            f.write("\n\n")
    
    print(f"📄 Sample data written to {sample_file}")
    print("✅ Dataset loading test completed successfully!")
    
    return len(texts), total_chars, len(all_tokens)

if __name__ == "__main__":
    try:
        num_conversations, total_chars, total_tokens = test_dataset_loading()
        print(f"\n🎉 Test Summary:")
        print(f"   ✅ Loaded {num_conversations} conversations")
        print(f"   ✅ Processed {total_chars:,} characters")
        print(f"   ✅ Generated {total_tokens:,} tokens")
        print(f"   ✅ All systems working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()