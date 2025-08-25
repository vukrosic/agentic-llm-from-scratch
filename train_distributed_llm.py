import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional
import warnings
import os
import pickle
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    accelerate_set_seed(seed)
    print(f"🌱 Set all seeds to {seed}")

@dataclass
class ModelConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 24  # Per GPU batch size
    num_epochs: int = 1  # Number of epochs through the full dataset

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01

    # Data parameters
    max_seq_len: int = 512
    data_fraction: float = 1.0  # Fraction of dataset to use (1.0 = full dataset, 0.04 = 1/26th)
    # Removed num_documents and max_tokens limits - using entire dataset
    
    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"
    vocab_size: Optional[int] = None
    
    # Checkpoint resumption
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint directory
    
    # Will be calculated dynamically
    max_steps: Optional[int] = None

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

# Note: torch.compile may need adjustment for distributed training
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # Add closure parameter
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

        return loss  # Return loss if closure was provided

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

def load_and_cache_data(config: ModelConfig, accelerator: Accelerator, cache_dir: str = "data_cache"):
    """Load and cache tokenized data from hermes_reasoning_tool_use dataset"""
    # Only process data on main process
    if accelerator.is_main_process:
        os.makedirs(cache_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()
    
    # Updated cache file name to reflect new dataset
    cache_file = f"{cache_dir}/hermes_reasoning_tokenized_data.pkl"

    # Check if cached data exists
    if os.path.exists(cache_file):
        accelerator.print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        accelerator.print(f"✅ Loaded {len(texts)} conversations, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    # Only main process downloads and processes
    if accelerator.is_main_process:
        accelerator.print(f"🔄 Processing hermes_reasoning_tool_use dataset (will cache for future use)")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load the hermes reasoning dataset (entire dataset - ~51k rows)
        dataset = load_dataset("interstellarninja/hermes_reasoning_tool_use", 
                              split="train", token=False)

        accelerator.print(f"📊 Dataset loaded: {len(dataset)} total conversations")

        # Extract text from conversations
        texts = []
        total_chars = 0
        
        accelerator.print("🔄 Extracting conversation texts...")
        for i, item in enumerate(tqdm(dataset, desc="Processing conversations", disable=not accelerator.is_local_main_process)):
            conversation = item["conversations"]
            extracted_text = extract_conversation_text(conversation)
            texts.append(extracted_text)
            total_chars += len(extracted_text)
            
            # Show progress every 10k items
            if (i + 1) % 10000 == 0:
                accelerator.print(f"  Processed {i + 1:,} conversations, {total_chars:,} characters")

        accelerator.print(f"📝 Extracted {len(texts)} conversations, {total_chars:,} total characters")

        # Tokenize all conversations
        accelerator.print("🔤 Tokenizing conversations...")
        all_tokens = []
        
        for i, text in enumerate(tqdm(texts, desc="Tokenizing", disable=not accelerator.is_local_main_process)):
            tokens = tokenizer.encode(text, add_special_tokens=True)  # Include special tokens for conversation structure
            all_tokens.extend(tokens)
            
            # Show progress every 10k items
            if (i + 1) % 10000 == 0:
                accelerator.print(f"  Tokenized {i + 1:,} conversations, {len(all_tokens):,} tokens so far")

        accelerator.print(f"🎯 Total tokens generated: {len(all_tokens):,}")
        config.vocab_size = tokenizer.vocab_size

        # Cache the processed data
        cached_data = {
            'texts': texts, 
            'tokenizer': tokenizer, 
            'tokens': all_tokens,
            'total_chars': total_chars,
            'num_conversations': len(texts)
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)

        accelerator.print(f"💾 Cached data to {cache_file}")
        
        # Create a sample data file
        sample_file = f"{cache_dir}/sample_conversations.txt"
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write("# Sample conversations from hermes_reasoning_tool_use dataset\n\n")
            f.write(f"Dataset Statistics:\n")
            f.write(f"- Total conversations: {len(texts):,}\n")
            f.write(f"- Total characters: {total_chars:,}\n")
            f.write(f"- Total tokens: {len(all_tokens):,}\n")
            f.write(f"- Vocabulary size: {config.vocab_size:,}\n\n")
            
            # Write first 5 conversations as samples
            for i in range(min(5, len(texts))):
                f.write(f"=== Conversation {i+1} ===\n")
                f.write(texts[i][:2000])  # First 2000 chars
                if len(texts[i]) > 2000:
                    f.write("\n... [truncated] ...\n")
                f.write("\n\n")
        
        accelerator.print(f"📄 Sample data written to {sample_file}")
        
        return texts, tokenizer, all_tokens
    
    # Wait for main process to finish
    accelerator.wait_for_everyone()
    
    # Load the cached data on all processes
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)
    
    texts = cached_data['texts']
    tokenizer = cached_data['tokenizer']
    tokens = cached_data['tokens']
    config.vocab_size = tokenizer.vocab_size
    
    return texts, tokenizer, tokens

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig, accelerator: Accelerator):
    """Evaluate model performance across all GPUs"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break

            with accelerator.autocast():
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            # Gather losses and metrics from all processes
            loss = accelerator.gather(loss)
            if accelerator.is_main_process:
                total_loss += loss.mean().item() * y.numel()
            
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            correct = (predictions == y).sum()
            correct = accelerator.gather(correct)
            if accelerator.is_main_process:
                total_correct += correct.sum().item()

    # Calculate metrics only on main process
    if accelerator.is_main_process:
        avg_loss = total_loss / (total_tokens * accelerator.num_processes)
        accuracy = total_correct / (total_tokens * accelerator.num_processes)
        perplexity = math.exp(min(avg_loss, 20))
    else:
        avg_loss = accuracy = perplexity = 0

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

def setup_muon_optimizer(model: nn.Module, config: ModelConfig, accelerator: Accelerator):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    accelerator.print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    accelerator.print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)

    return [muon_optimizer, adamw_optimizer]

def train_model(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, accelerator: Accelerator):
    """Train the model with Muon optimizer using Accelerate"""
    accelerator.print(f"\n🚀 Training Small model with Muon optimizer on {accelerator.num_processes} GPU(s)")

    # Initialize model
    set_seed(42)
    model = MinimalLLM(config)

    total_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"  📊 Total parameters: {total_params:,}")

    # Setup optimizers
    optimizers = setup_muon_optimizer(model, config, accelerator)

    # Learning rate schedule
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    # Prepare everything with accelerator
    model, *optimizers = accelerator.prepare(model, *optimizers)
    train_loader, val_loader, test_loader = accelerator.prepare(train_loader, val_loader, test_loader)
    
    # Prepare schedulers separately (they don't need prepare)
    for i, scheduler in enumerate(schedulers):
        schedulers[i] = accelerator.prepare(scheduler)

    # Checkpoint resumption
    starting_step = 0
    if config.resume_from_checkpoint:
        if os.path.exists(config.resume_from_checkpoint):
            accelerator.print(f"🔄 Resuming training from {config.resume_from_checkpoint}")
            accelerator.load_state(config.resume_from_checkpoint)
            
            # Extract step number from checkpoint directory name if possible
            checkpoint_name = os.path.basename(config.resume_from_checkpoint)
            if "step_" in checkpoint_name:
                try:
                    starting_step = int(checkpoint_name.split("step_")[1])
                    accelerator.print(f"   Resuming from step {starting_step}")
                except ValueError:
                    accelerator.print(f"   Could not extract step number, starting from 0")
            else:
                accelerator.print(f"   Resuming training (step number unknown)")
        else:
            accelerator.print(f"⚠️  Checkpoint directory {config.resume_from_checkpoint} not found, starting fresh training")
    
    # Auto-resume from latest checkpoint if available
    elif os.path.exists("checkpoint_best"):
        accelerator.print(f"🔄 Auto-resuming from checkpoint_best")
        accelerator.load_state("checkpoint_best")
        accelerator.print(f"   Loaded best checkpoint")

    # Training loop
    model.train()
    step = starting_step
    start_time = time.time()
    best_val_loss = float('inf')

    # Progress bar only on main process
    pbar = tqdm(total=config.max_steps, desc="Training", disable=not accelerator.is_local_main_process)

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            # Use accelerator's accumulation context
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

                # Use accelerator's backward
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)

                # Optimizer steps
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Scheduler steps
                for scheduler in schedulers:
                    scheduler.step()

            # Logging (only on main process)
            if step % 100 == 0 and accelerator.is_main_process:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = loss.item()
                    perplexity = math.exp(min(current_loss, 20))

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{optimizers[0].param_groups[0]["lr"]:.2e}'
                })

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config, accelerator)
                if accelerator.is_main_process:
                    accelerator.print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                          f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                          f"Val PPL: {eval_metrics['val_perplexity']:.2f}")

                    if eval_metrics['val_loss'] < best_val_loss:
                        best_val_loss = eval_metrics['val_loss']
                        # Save best model
                        accelerator.save_state(output_dir=f"checkpoint_best")
                        accelerator.print(f"   🏆 New best model saved!")
                    
                    # Save periodic checkpoint for resumption (every 10k steps)
                    if step % 10000 == 0:
                        accelerator.save_state(output_dir=f"checkpoint_step_{step}")
                        accelerator.print(f"   💾 Checkpoint saved at step {step}")

            step += 1
            if step % 100 == 0 and accelerator.is_local_main_process:
                pbar.update(100)

    if accelerator.is_local_main_process:
        pbar.close()

    training_time = time.time() - start_time
    accelerator.print(f"  ⏱️ Training completed in {training_time:.1f} seconds")

    # Final evaluation on validation set
    final_eval = evaluate_model(model, val_loader, config, accelerator)
    if accelerator.is_main_process:
        accelerator.print(f"  📊 Final Validation - Loss: {final_eval['val_loss']:.4f}, "
              f"Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")

    # Final test evaluation on held-out test set (never seen during training)
    test_eval = evaluate_model(model, test_loader, config, accelerator)
    if accelerator.is_main_process:
        accelerator.print(f"  🏆 Final Test (Unseen) - Loss: {test_eval['val_loss']:.4f}, "
              f"Acc: {test_eval['val_accuracy']:.4f}, PPL: {test_eval['val_perplexity']:.2f}")

    return model, final_eval, test_eval

def main():
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="fp16",  # or "bf16" for better numerical stability
        log_with="tensorboard",  # Optional: add logging
        project_dir="./outputs"
    )

    # Check system
    accelerator.print(f"🔍 Using {accelerator.num_processes} GPU(s)")
    if torch.cuda.is_available():
        accelerator.print(f"GPU: {torch.cuda.get_device_name()}")
        accelerator.print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Set seed
    set_seed(42)

    # Create config for Small model
    config = ModelConfig()
    config.mixed_precision = accelerator.mixed_precision
    
    # QUICK TEST CONFIGURATION - Uncomment for testing:
    config.max_steps = None  # Manual override for testing
    config.data_fraction = 0.04  # Use 1/26th of the data (as requested)
    accelerator.print(f"🗺️ TEST MODE: Using {config.data_fraction:.1%} of data, max_steps = {config.max_steps}")
    
    # Check for resume checkpoint from environment variable (used by resume_training.py)
    if os.environ.get('RESUME_FROM_CHECKPOINT'):
        config.resume_from_checkpoint = os.environ['RESUME_FROM_CHECKPOINT']
        accelerator.print(f"🔄 Environment variable set - will resume from: {config.resume_from_checkpoint}")
    
    # Uncomment to resume from a specific checkpoint manually:
    # config.resume_from_checkpoint = "checkpoint_step_1500"  # or "checkpoint_best" or "checkpoint_final"
    
    if config.resume_from_checkpoint:
        accelerator.print(f"🔄 Will attempt to resume from: {config.resume_from_checkpoint}")
    
    accelerator.print(f"\n📋 Model Configuration:")
    accelerator.print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
    accelerator.print(f"   Training: {config.num_epochs} epochs, batch size {config.batch_size} per GPU")
    accelerator.print(f"   Data: hermes_reasoning_tool_use dataset, seq_len {config.max_seq_len}")

    # Load data
    texts, tokenizer, tokens = load_and_cache_data(config, accelerator)
    
    # Apply data reduction if specified
    if config.data_fraction < 1.0:
        original_token_count = len(tokens)
        reduced_token_count = int(len(tokens) * config.data_fraction)
        tokens = tokens[:reduced_token_count]
        accelerator.print(f"🔥 Data reduction: Using {config.data_fraction:.1%} of dataset")
        accelerator.print(f"   Original tokens: {original_token_count:,}")
        accelerator.print(f"   Reduced tokens: {len(tokens):,}")
    
    dataset = TextTokenDataset(tokens, config.max_seq_len)
    
    accelerator.print(f"📊 Dataset loaded: {len(texts):,} conversations, {len(tokens):,} total tokens")

    # Train/val/test split (89/10/1) - minimal test set for final evaluation
    test_size = max(100, len(dataset) // 100)  # 1% or minimum 100 samples
    val_size = len(dataset) // 10  # 10%
    train_size = len(dataset) - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # Calculate max_steps based on dataset size to ensure full dataset training
    # BUT only if max_steps is not already manually set
    if config.max_steps is None:
        effective_batch_size = config.batch_size * accelerator.num_processes * config.gradient_accumulation_steps
        steps_per_epoch = len(train_dataset) // effective_batch_size
        config.max_steps = steps_per_epoch * config.num_epochs
        accelerator.print(f"📊 Training Configuration (Dynamic):")
    else:
        accelerator.print(f"📊 Training Configuration (Manual max_steps):")
        accelerator.print(f"   Manual max_steps setting: {config.max_steps}")
    
    effective_batch_size = config.batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    steps_per_epoch = len(train_dataset) // effective_batch_size
    
    accelerator.print(f"   Steps per epoch: {steps_per_epoch:,}")
    accelerator.print(f"   Total epochs: {config.num_epochs}")
    accelerator.print(f"   Total training steps: {config.max_steps:,}")
    accelerator.print(f"   Effective batch size: {effective_batch_size}")

    # Create dataloaders (Accelerate will handle distribution)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    accelerator.print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")

    # Train model
    start_time = time.time()
    model, final_metrics, test_metrics = train_model(config, train_loader, val_loader, test_loader, accelerator)
    total_time = time.time() - start_time

    if accelerator.is_main_process:
        accelerator.print(f"\n🎉 TRAINING COMPLETED!")
        accelerator.print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        accelerator.print(f"🏆 Final Results:")
        accelerator.print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
        accelerator.print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
        accelerator.print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
        accelerator.print(f"\n🎯 Test Set (Unseen Data):")
        accelerator.print(f"   Test Loss: {test_metrics['val_loss']:.4f}")
        accelerator.print(f"   Test Accuracy: {test_metrics['val_accuracy']:.4f}")
        accelerator.print(f"   Test Perplexity: {test_metrics['val_perplexity']:.2f}")

        # Save final model
        accelerator.save_state(output_dir="checkpoint_final")

if __name__ == "__main__":
    main()