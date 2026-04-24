# Python Code generator using LLM Training Pipeline

## 1. Overview
This project implements an end-to-end pipeline for training a GPT-style language model for Python code generation. It covers data collection, tokenizer training, model pretraining, and distributed training using Hugging Face Accelerate.

The system is designed to handle large-scale datasets (100GB+) using streaming and memory-efficient processing.

---

## 2. Theory

The model is trained using **causal language modeling (CLM)**, where the objective is to predict the next token given previous tokens:

```math
P(x_t | x_1, x_2, ..., x_{t-1})
```

This is the standard objective used in GPT-style decoder-only transformers.

The pipeline includes:
- Byte-level BPE tokenizer trained on Python code
- GPT-style transformer decoder architecture
- Sequence packing for efficient training on long streams of code
- Perplexity-based evaluation

---

## 3. Tools and Technologies

- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets (streaming API)
- Hugging Face Accelerate (distributed training)
- TensorBoard (logging)
- Weights & Biases (experiment tracking)
- Google BigQuery (GitHub dataset extraction)

---

## 4. Data Collection

Data is collected from:
1. Google BigQuery GitHub Python dataset (filtered for `.py` files)
2. Hugging Face `transformersbook/codeparrot` dataset (183GB+)

Key techniques:
- Streaming data loading (no full dataset storage required)
- Memory-efficient iteration over large-scale code corpus
- Dataset splitting into train and validation sets

---

## 5. Tokenizer Training

A custom Byte-Level BPE tokenizer is trained on Python code to improve code-specific representation.

Steps:
1. Load streaming dataset of Python code
2. Build batch iterator over raw code files
3. Train tokenizer using `train_new_from_iterator`
4. Evaluate tokenization quality on Python keywords and syntax
5. Push tokenizer to Hugging Face Hub

The tokenizer is optimized to reduce fragmentation of programming keywords and improve code representation efficiency.

---

## 6. Model Architecture and Training

A GPT-style decoder-only transformer is initialized using Hugging Face configurations:

- GPT-2 / GPT-2 XL architecture
- Configurable vocabulary size from custom tokenizer
- Causal language modeling objective

Training setup:
1. Initialize model from config or pretrained checkpoint
2. Build constant-length dataset with sequence packing
3. Use gradient accumulation for large effective batch size
4. Train using custom PyTorch loop or Accelerate framework
5. Apply learning rate scheduling (cosine decay)
6. Evaluate using loss and perplexity

---

## 7. Distributed Training (Accelerate)

Training is scaled using Hugging Face Accelerate:

- Multi-GPU / multi-process support
- Mixed precision training
- Gradient synchronization across devices
- Efficient distributed dataloading
- Checkpoint saving and resuming
- Integration with Hugging Face Hub

Training loop includes:
- Forward pass with causal LM loss
- Backpropagation with gradient accumulation
- Gradient clipping
- Optimizer step and LR scheduler update
- Periodic evaluation and logging

---

## 8. Evaluation

The model is evaluated using:
- Cross-entropy loss
- Perplexity (exp(loss))

Lower perplexity indicates better language modeling performance on unseen code.

Validation is performed on held-out streaming dataset.

---

## 9. Results

The final trained model:
- Generates syntactically valid Python code
- Completes partial functions and scripts
- Demonstrates improved token efficiency compared to generic tokenizers
- Supports code synthesis tasks such as function generation and API usage prediction

Example capabilities:
- Function completion
- Code translation (NumPy / PyTorch / sklearn style)
- HTML parsing utilities
- Algorithmic code generation

The model can be deployed via Hugging Face pipeline for inference:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="your-trained-model")
generator("def fibonacci(n):")
```
