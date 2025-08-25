# Agentic LLM from Scratch

New video: How To Code & Train Agentic LLM

- YouTube - https://youtu.be/ij8Ru4jxRQY

- Bilibili (中文) - https://www.bilibili.com/video/BV15WvFz8E2R/

A minimalist implementation for training large language models (LLMs) with agentic capabilities from scratch using distributed training. This project focuses on the pretraining phase of an agentic LLM using the Hermes reasoning dataset.

## 🎯 Project Overview

This project demonstrates how to build and train an LLM with agent-like behavior, supporting autonomous decision-making and reasoning capabilities. The implementation uses distributed training for scalability and is designed for educational and research purposes.

### Key Features

- ✅ **Distributed Training**: Automatic GPU detection and distributed training via PyTorch Accelerate
- ✅ **Agentic Dataset**: Trained on `interstellarninja/hermes_reasoning_tool_use` dataset with reasoning traces
- ✅ **Custom Architecture**: Transformer model with rotary embeddings and Muon optimizer
- ✅ **Checkpoint Management**: Automatic saving and resumption from checkpoints
- ✅ **Inference Engine**: Complete inference script for text generation
- ✅ **Configurable Training**: Support for data reduction and manual step control

## 🏗️ Architecture

### Model Architecture
- **Size**: 384d model, 6 layers, 8 heads
- **Parameters**: ~50M parameters
- **Context Length**: 512 tokens
- **Architecture**: Transformer with RMSNorm, SiLU activation, and rotary embeddings
- **Optimizer**: Hybrid Muon + AdamW optimization

### Training Features
- **Dataset**: Hermes reasoning dataset with conversation formatting
- **Tokenizer**: SmolLM-135M tokenizer
- **Mixed Precision**: FP16 training for efficiency
- **Data Parallel**: Automatic distributed training across available GPUs

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support
- Hugging Face Transformers
- Accelerate
- Datasets

### Installation
```bash
git clone <repository-url>
cd agentic-llm-from-scratch
pip install torch transformers accelerate datasets tqdm
```

### Basic Training
```bash
# Single GPU training
python train_distributed_llm.py

# Multi-GPU training
torchrun --nproc_per_node=2 train_distributed_llm.py

# Quick test with reduced data
# Edit train_distributed_llm.py and set:
# config.max_steps = 300
# config.data_fraction = 0.04
```

### Inference
```bash
# Interactive chat mode
python inference.py --interactive

# Single prompt
python inference.py

# Use specific checkpoint
python inference.py checkpoint_step_10000 --interactive
```

## 📊 Training Configuration

### Default Settings
```python
ModelConfig(
    d_model=384,          # Model dimension
    n_heads=8,            # Attention heads
    n_layers=6,           # Transformer layers
    d_ff=1536,            # Feed-forward dimension
    batch_size=24,        # Per-GPU batch size
    num_epochs=1,         # Training epochs
    max_seq_len=512,      # Sequence length
    data_fraction=1.0,    # Fraction of dataset (1.0 = full)
)
```

### Quick Test Configuration
For rapid iteration and testing:
```python
config.max_steps = 300        # Limit training steps
config.data_fraction = 0.04   # Use 4% of data (26x faster)
```

## 💾 Checkpoint Management

### Automatic Saving
- **Best Model**: `checkpoint_best` (lowest validation loss)
- **Final Model**: `checkpoint_final` (end of training)
- **Periodic**: `checkpoint_step_N` (every 10,000 steps)

### Resuming Training
```bash
# Auto-resume from latest checkpoint
python resume_training.py

# Resume from specific checkpoint
python resume_training.py checkpoint_step_10000

# Manual resumption (edit train_distributed_llm.py)
config.resume_from_checkpoint = "checkpoint_best"
```

## 🎮 Inference Usage

### Interactive Chat
```bash
python inference.py --interactive --max_tokens 300 --temperature 0.8
```

Example conversation:
```
🧑 You: What is the word frequency of "hello world hello"?

🤖 Assistant: <think>
The user wants me to calculate word frequency for "hello world hello". 
I need to count each word:
- "hello" appears 2 times
- "world" appears 1 time
</think>

The word frequency for "hello world hello" is:
- hello: 2
- world: 1
```

### Single Prompt
```bash
python inference.py
# Enter prompt when asked
```

### Parameters
- `--max_tokens`: Maximum tokens to generate (default: 200)
- `--temperature`: Sampling temperature 0.1-1.0 (default: 0.8)
- `--interactive`: Enable chat mode

## 📁 Project Structure

```
agentic-llm-from-scratch/
├── train_distributed_llm.py     # Main training script
├── inference.py                 # Inference engine
├── resume_training.py           # Checkpoint resumption utility
├── test_data_loading.py         # Data loading test script
├── data_cache/                  # Cached dataset files
│   ├── hermes_reasoning_tokenized_data.pkl
│   └── sample_conversations.txt
├── checkpoint_best/             # Best model checkpoint
├── checkpoint_final/            # Final model checkpoint
├── checkpoint_step_N/           # Periodic checkpoints
└── README.md                    # This file
```

## 🔧 Advanced Usage

### Custom Training
Edit `train_distributed_llm.py` main function:
```python
# Custom configuration
config.max_steps = 1000           # Manual step limit
config.data_fraction = 0.1        # Use 10% of data
config.batch_size = 32            # Larger batch size
config.muon_lr = 0.02             # Learning rate
```

### Multi-GPU Training
```bash
# Use all available GPUs
python train_distributed_llm.py

# Specify GPU count
torchrun --nproc_per_node=4 train_distributed_llm.py
```

### Data Processing
The training script automatically:
1. Downloads the Hermes reasoning dataset
2. Formats conversations with role tags (`<|system|>`, `<|user|>`, `<|assistant|>`)
3. Tokenizes using SmolLM tokenizer
4. Caches processed data for future runs
5. Creates sample files for verification

## 📈 Training Output

Expected training logs:
```
🔍 Using 1 GPU(s)
📋 Model Configuration:
   Architecture: 384d, 6L, 8H, 1536ff
   Training: 1 epochs (15,234 steps), batch size 24 per GPU

📊 Dataset loaded: 51,020 conversations, 79,234,567 total tokens
📊 Training Configuration (Manual max_steps):
   Manual max_steps setting: 300
   Steps per epoch: 15,234
   Total epochs: 1
   Total training steps: 300
   Effective batch size: 96

Training: 100%|████| 300/300 [00:18<00:00, 16.19it/s, loss=1.1907, acc=0.778, ppl=3.98]
  📊 Final Validation - Loss: 1.3812, Acc: 0.7562, PPL: 3.98
  🏆 Final Test (Unseen) - Loss: 1.4123, Acc: 0.7401, PPL: 4.11
```

## 🎯 Model Capabilities

The trained model should be capable of:
- **Reasoning**: Step-by-step thinking with `<think>` tags
- **Tool Usage**: Understanding function calling patterns
- **Conversation**: Multi-turn dialogue capabilities
- **Problem Solving**: Mathematical and logical reasoning

### Example Capabilities
```
User: Calculate the age of someone born in 1990
Assistant: <think>
To calculate age, I need to subtract birth year from current year.
Current year is 2024, birth year is 1990.
Age = 2024 - 1990 = 34
</think>

If someone was born in 1990, they would be 34 years old in 2024.
```

## 🔬 Technical Details

### Optimizer
- **Muon**: For 2D parameters (linear layers)
- **AdamW**: For 1D parameters (embeddings, norms)
- **Learning Rate**: Cosine schedule with warmup

### Memory Optimization
- **Mixed Precision**: FP16 training
- **Gradient Accumulation**: 4 steps
- **Data Caching**: Preprocessed dataset caching

### Evaluation
- **Train/Val/Test Split**: 89%/10%/1%
- **Evaluation Frequency**: Every 500 steps
- **Metrics**: Loss, accuracy, perplexity

## 🐛 Troubleshooting

### Common Issues

**Device Mismatch Error**:
```bash
RuntimeError: Expected all tensors to be on the same device
```
- **Solution**: All data loaders are properly prepared with Accelerator

**Out of Memory**:
- Reduce `batch_size` or `data_fraction`
- Use gradient checkpointing (add to model)

**Checkpoint Not Found**:
```bash
python inference.py
# Lists available checkpoints
```

**Training Too Slow**:
- Set `config.data_fraction = 0.1` for 10x faster training
- Use `config.max_steps = 1000` to limit training

## 📚 References

- **Dataset**: [interstellarninja/hermes_reasoning_tool_use](https://huggingface.co/datasets/interstellarninja/hermes_reasoning_tool_use)
- **Tokenizer**: [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
- **Architecture**: Transformer with rotary embeddings
- **Optimizer**: [Muon Optimizer](https://github.com/KellerJordan/Muon)

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different architectures
- Try different datasets
- Implement new optimization techniques
- Add evaluation metrics

## 📄 License

This project is for educational and research purposes. Please respect the licenses of the underlying datasets and models used.

---

**Happy Training! 🚀**

For questions or issues, please check the troubleshooting section or examine the training logs for detailed error information.