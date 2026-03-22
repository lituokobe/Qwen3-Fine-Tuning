## Unsloth
an open‑source library designed to make LLM fine‑tuning:
- Faster (2×–5× speedup)
- Cheaper (less VRAM needed)
- More memory‑efficient 
- Easier to use

It focuses on:
- LoRA fine‑tuning 
- QLoRA (quantized LoRA)
- Training large models on consumer GPUs (3090, 4090, etc.)

## Model precision:
- FP16 is NOT quantized. It’s a normal, high‑precision format.
- BF16 is also NOT quantized. It’s just a different 16‑bit floating‑point format. 
  - Larger exponent range than FP16, more stable during training
  - Preferred for fine‑tuning
  - Not supported by all GPUs 
- If a model is labeled AWQ or GPTQ, it is almost always INT4.  
  - AWQ (Activation‑aware Weight Quantization)
  - GPTQ is Another quantization method. Slightly more accuracy loss than AWQ.

## Instruct models
trained to follow human instructions:
- Chat 
- Classification 
- Tool use 
- Coding 
- Math 
- Structured output 
- Alignment with human preference

Instruct models have the same architecture, same number of layers, same parameters, same attention heads, same context length.

Their difference from the base model exists only in the training:
- They take the base pretrained model 
- Then apply supervised fine‑tuning (SFT) on instruction‑following datasets 
- Then apply alignment (RLHF, DPO, or similar)
- Sometimes apply safety tuning 
- Sometimes apply chat formatting tuning

Reasoning models can be considered an enhanced form of instruct models in practice. But conceptually:
- Instruct = alignment capability 
- Reasoning = cognitive depth capability 
- They overlap, but are not the same concept.
- All reasoning models start as instruct models: A base model → then instruction‑tuned → then reasoning‑tuned (the extra step)

You can think of today’s flagship models (Qwen‑Max, GPT‑5.2, Claude‑3.7, Gemini‑2.0 Ultra, etc.) as “Instruct‑style” models, even if companies don’t always use that word anymore.
They are “everything models”: Instruct + Reasoning + Tool Use + Safety + Multimodal + Long‑context

Still, open‑source communities follow the classic two‑step pipeline:
1. Pretrain a base model 
2. Release an instruction‑tuned version

Examples:
- Qwen3‑4B → Qwen3‑4B‑Instruct
- Llama‑3‑8B → Llama‑3‑8B‑Instruct
- Mistral‑7B → Mistral‑7B‑Instruct
- Gemma‑2B → Gemma‑2B‑Instruct

## LoRA
LoRA is the most popular fine‑tuning method because it hits the perfect balance of efficiency, accuracy, and practicality.
LoRA = Modular Fine‑Tuning (like plugins for your model). It keeps the base model 100% untouched. This gives you:

- Modularity
- Reversibility
- Composability 
- Version control
- Safe experimentation

data generation model: 
- 40%: GLM-4.7, flagship language model from Zhipu, deliver both diversity and accuracy
- 20%: GPT-4.1, most advance non-reasoning model from OpenAI - easy to generate repetitive patterns, also some non accuracy that needs manual check, but sometimes have good ones
- 20%: Gemini, 
- 10%: Qwen-max,
- 5%: GPT-5,
- 5%: Qwen-Plus