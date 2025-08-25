#!/usr/bin/env python3
"""
Inference script for the trained agentic LLM.
Usage:
    python inference.py [checkpoint_path]
    
Example:
    python inference.py checkpoint_best
    python inference.py checkpoint_final
    python inference.py checkpoint_step_300
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
from transformers import AutoTokenizer
from accelerate import Accelerator
from dataclasses import dataclass
from typing import Optional
import argparse

# Import model classes from training script
@dataclass
class ModelConfig:
    # Model architecture (must match training config)
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    
    # Data parameters
    max_seq_len: int = 512
    dropout: float = 0.1
    
    # Technical
    vocab_size: Optional[int] = None

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        Q = self.rotary(Q)
        K = self.rotary(K)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class MinimalLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.max_seq_len, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Tie weights
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits

class InferenceEngine:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔍 Using device: {self.device}")
        
        # Load tokenizer
        print("📥 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create model config
        self.config = ModelConfig()
        self.config.vocab_size = self.tokenizer.vocab_size
        
        # Initialize model
        print("🤖 Initializing model...")
        self.model = MinimalLLM(self.config)
        
        # Load checkpoint
        print(f"📦 Loading checkpoint from {checkpoint_path}...")
        self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        print("✅ Model loaded and ready for inference!")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint using Accelerator"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_path} not found!")
        
        # Use Accelerator to load the checkpoint properly
        accelerator = Accelerator()
        
        # Prepare model
        self.model = accelerator.prepare(self.model)
        
        # Load state
        accelerator.load_state(checkpoint_path)
        
        print(f"   ✅ Loaded checkpoint from {checkpoint_path}")

    def format_conversation(self, messages):
        """Format messages into the training format"""
        formatted_text = ""
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'system':
                formatted_text += f"<|system|>\n{content}\n\n"
            elif role == 'user':
                formatted_text += f"<|user|>\n{content}\n\n"
            elif role == 'assistant':
                formatted_text += f"<|assistant|>\n{content}\n\n"
        
        return formatted_text

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8, top_p: float = 0.9):
        """Generate text from prompt"""
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        print(f"🔤 Input tokens: {input_ids.shape[1]}")
        print(f"📝 Generating {max_new_tokens} new tokens...")
        
        generated = input_ids.clone()
        
        for i in range(max_new_tokens):
            # Get model output
            with torch.cuda.amp.autocast():
                logits = self.model(generated)
            
            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Check for end of sequence
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Truncate if too long
            if generated.shape[1] > self.config.max_seq_len:
                generated = generated[:, -self.config.max_seq_len:]
        
        # Decode response
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=False)
        new_text = self.tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
        
        return new_text, generated_text

    def chat(self, messages, max_new_tokens: int = 200, temperature: float = 0.8):
        """Chat interface using conversation format"""
        # Format the conversation
        formatted_prompt = self.format_conversation(messages)
        formatted_prompt += "<|assistant|>\n"  # Prompt for assistant response
        
        print(f"💬 Formatted prompt:\n{formatted_prompt}")
        print("=" * 50)
        
        # Generate response
        response, full_text = self.generate(
            formatted_prompt, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature
        )
        
        return response.strip()

def main():
    parser = argparse.ArgumentParser(description="Inference script for trained agentic LLM")
    parser.add_argument("checkpoint", nargs="?", default="checkpoint_best", 
                       help="Path to checkpoint directory (default: checkpoint_best)")
    parser.add_argument("--max_tokens", type=int, default=200,
                       help="Maximum new tokens to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature (default: 0.8)")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive chat mode")
    
    args = parser.parse_args()
    
    try:
        # Initialize inference engine
        engine = InferenceEngine(args.checkpoint)
        
        if args.interactive:
            print("\n🎯 Interactive Chat Mode")
            print("Type 'quit' to exit, 'clear' to clear conversation history")
            print("=" * 50)
            
            conversation = []
            
            while True:
                user_input = input("\n🧑 You: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'clear':
                    conversation = []
                    print("💫 Conversation cleared!")
                    continue
                elif not user_input:
                    continue
                
                # Add user message
                conversation.append({"role": "user", "content": user_input})
                
                # Generate response
                print("\n🤖 Assistant: ", end="", flush=True)
                response = engine.chat(conversation, args.max_tokens, args.temperature)
                print(response)
                
                # Add assistant response to conversation
                conversation.append({"role": "assistant", "content": response})
        
        else:
            # Single prompt mode
            print("\n🎯 Single Prompt Mode")
            print("Enter your prompt (or 'quit' to exit):")
            
            user_input = input("\n🧑 Prompt: ").strip()
            if user_input and user_input.lower() != 'quit':
                messages = [{"role": "user", "content": user_input}]
                response = engine.chat(messages, args.max_tokens, args.temperature)
                print(f"\n🤖 Response: {response}")
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nAvailable checkpoints:")
        for item in os.listdir('.'):
            if item.startswith('checkpoint_') and os.path.isdir(item):
                print(f"  - {item}")
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()