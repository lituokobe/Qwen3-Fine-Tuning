# PEFT Practice with Hugging Face APIs

This repository is a hands-on exploration of **Parameter-Efficient Fine-Tuning (PEFT)** techniques using Hugging Face's ecosystem. It serves as a personal sandbox for experimenting with various strategies to adapt large language models efficiently.

## Techniques Covered

The following PEFT methods are implemented and tested:

- **BitFit** – Fine-tuning only bias terms
- **Prompt Tuning** – Learnable prompts injected at the input layer
- **P-Tuning** – Task-specific prefix embeddings generated via MLPs
- **Prefix Tuning** – Layer-wise prefix vectors prepended to attention keys/values
- **LoRA (Low-Rank Adaptation)** – Injecting low-rank matrices into weight updates
- **IA³ (Infused Adapter Adjustment)** – Scaling activations with learned vectors
- **Advanced Adapter Fusion** – Activating/deactivating/switching adapters
- **PEFT Model Merging** – Combining fine-tuned adapters with existing models

## Foundation Model

All experiments are based on:

**Langboat 1b4-zh**  
A lightweight Chinese-centric LLM designed for reasoning tasks. Its compact architecture makes it ideal for testing PEFT strategies with minimal compute overhead.

## Goals

- Understand the internal mechanics of each PEFT method
- Compare performance and efficiency trade-offs
- Explore adapter merging and multi-adapter setups
- Build reproducible workflows for scalable fine-tuning

## Requirements

- Python ≥ 3.8  
- `transformers`, `peft`, `datasets`, `accelerate`  
- GPU recommended for training

## Notes

This repository is intended for educational and experimental purposes. Contributions, suggestions, and discussions are welcome!

