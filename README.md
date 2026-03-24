# Qwen3 LoRA Fine-Tuning for Intent Classification

<p>
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python Version">
    </a>
    <a href="https://pytorch.org/">
        <img src="https://img.shields.io/badge/Framework-PyTorch-orange" alt="Framework">
    </a>
    <a href="https://huggingface.co/Qwen">
        <img src="https://img.shields.io/badge/Model-Qwen3%20%7C%20Qwen3.5-9cf" alt="Model Family">
    </a>
    <a href="https://github.com/microsoft/LoRA">
        <img src="https://img.shields.io/badge/Technique-LoRA-green" alt="Technique">
    </a>
    <a href="LICENSE">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache 2.0 License">
    </a>
</p>

This repository provides a complete pipeline for fine-tuning compact LLMs for **intent classification in real-time conversational systems**.

The fine-tuned model achieves:
- **93.5% accuracy** on a test set with diverse customization
- **0.37s inference** latency on production GPUs

## 📌 1. Background

My company is a martech startup that leverages technology to engage with clients at commercial events (car shows, renovation fairs, etc.).
Recently, we are pioneering [using AI provide customer services](https://github.com/lituokobe/AI-CS-Agent) via phone calls.  

To be specific, business managers build conversational flows using a node-based GUI, where each node represents a stage in the conversation.
At each node, the system must **identify the user's intent** in real time to determine the next action.

This repository focuses on solving that intent classification problem.

## 🚧 2. Challenges:
- **Dynamic Intent Space**
  - Intent categories are user-defined and dynamic
  - They vary across industries and use cases
  - Requires strong generalization under constrained supervision
- Real-Time Constraints
  - Phone recipients are less patient when waiting for replies
  - Voice pipeline includes ASR + LLM + TTS
  - Total latency target: < 1 second
  - Intent classification budget: < 0.5 seconds
- Cost Constraints
  - Frequent API calls to flagship models (e.g., Qwen3.5) are expensive
  - Need a low-cost, scalable local solution

## 🚀 3. Solution

I decided to fine-tune small open-source LLMs (≈4B parameters) using LoRA, achieving a strong balance between:
- accuracy 
- speed 
- cost 
- controllability

### Comparison

| Approach                                  | Cost                      | Latency                          | Control & Reliability        | Performance                 |
|:------------------------------------------|:--------------------------|:---------------------------------|:-----------------------------|:----------------------------|
| **Commercial API Calls**                  | High (per-token cost)     | Medium–High (network bottleneck) | Low (dependent on 3rd party) | High                        |
| **Large Local Model (e.g., 70B)**         | Very High (hardware cost) | Low (with high-end GPUs)         | Full                         | High                        |
| **Small Local Model (e.g., 4B)**          | Low                       | Low (<0.4s)                      | Full                         | Moderate (70%+)             |
| **Fine-Tuned Small Model (This Project)** | **Low**                   | **Low (<0.4s)**                  | **Full**                     | **Production-Ready (90%+)** |

### Key Benefits
- **🏎️ Performance & Efficiency:** With private deployment, inference time is under **0.5 seconds**, providing a seamless user experience.
- **🎯 Domain Adaptation:** Trained data specifically generated for customer engagement (in the company's business).
- **🔒 Full Control & Data Privacy:** Complete ownership of the model and infrastructure ensures reliable maintenance, easy debugging, and that all customer data remains in‑house.
- **💰 Cost Efficiency:** No per-token API cost, only flat cost of GPU cloud server.

## 🤖 4. Model Selection

I experimented two popular, open‑source models from the Qwen family as our base models:

- [**`Qwen3-4B-Instruct-2507`**](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) – The most popular open‑sourced instruct model from Qwen. Its small size and lack of built‑in reasoning make it ideal for a fast, focused classification task.
- [**`Qwen3.5-4B`**](https://huggingface.co/Qwen/Qwen3.5-4B) – Just released in March 2026, the flagship open‑sourced multimodal model from Qwen, with reasoning turned off. Not used in final results due to inference instability (reasoning mode).

## 📊 5. Dataset

### 5.1 Design Strategy

#### Optimized Input Architecture
To maximize efficiency for small-parameter models and minimize token overhead, this project utilizes a custom-engineered input format rather than the standard LangChain message list.

- **Token Efficiency:** By concatenating the system prompt and chat history into a single string, I eliminate the metadata overhead associated with traditional message objects. 
- **Positional Relevancy:** The input is structured to place the last user input, chat history, and intention options in close proximity. This leverages the Transformer architecture's attention mechanism, ensuring the model prioritizes the most relevant context for intent identification.
- **Performance-First Design:** This layout reduces "distraction" within the context window, allowing smaller models to perform with the accuracy of much larger counterparts.

#### Data-Centric Foundation
High-quality, diverse data is the cornerstone of this project. Our dataset was constructed through a rigorous multi-stage process designed to mirror the complexity of real-world customer interactions. 
Training samples are formatted to reflect our optimized input structure, with broad variations in prompt settings and intent possibilities to ensure robust generalization.

### 5.2. Data Generation
A total of **10,000+ training samples in Chinese** were generated using a mixture of state‑of‑the‑art models. This blend ensures both high accuracy and diversity, including edge cases.

- **50% [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7) (Zhipu)** – Used as the primary generator for its high accuracy and ability to produce diverse, in‑domain customer engagement conversation in Chinese.
- **25% [GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1) (OpenAI)** – Crucial for generating challenging edge cases, including rude, vulgar, or disrespectful language, which models from China with stricter safety filters often avoid.
- **15% [Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) (Moonshot AI)** – Adds further linguistic diversity. All outputs from this model undergo manual review for quality assurance and it is relative weaker in logic although strong in creativity.
- **5% [Gemini-3](https://deepmind.google/models/gemini/) (Google)** – Another non‑Chinese model used to add diversity, particularly for generating content related to culturally sensitive topics.
- **5% [Qwen-Max](https://modelstudio.console.alibabacloud.com/ap-southeast-1?tab=doc#/doc/?type=model&url=2840914_2&modelId=qwen3-max) (Alibaba)** – Another high‑precision model for Chinese language, though its outputs tend to follow predictable patterns, reducing overall dataset diversity.

### 5.3. Quality Assurance
- **Initial Seed Data:** 100 manually crafted samples were created based on discussions with business managers to cover both common and niche scenarios.
- **First Round Generation:** 1,000+ samples were generated with above models, manually reviewd, and then used for a quick training for process validation.
- **Bulk Generation:** 9,000+ samples were eventually generated for the final training, following a plan that covers all the possible cases: different intention priorities, commmon/niche intentions, positive/negative intentions, excited/neutral/rude attitudes.
- **Manual Review:** **40% of all generated data** was manually checked and corrected to ensure accuracy and relevance.

👉[View the dataset here.](https://huggingface.co/datasets/lituokobe/LoRA-Samples-Intention-Classifier)

## ⚙️ 6. Training

- **Technique:** **LoRA (Low‑Rank Adaptation)** was chosen for its efficiency and modularity. It allows us to fine‑tune the model on a consumer‑grade GPU without catastrophic forgetting, and the resulting adapter can be easily merged or swapped.
- **Training Framework:** 
  - [Huggingface Transformers](https://huggingface.co/docs/transformers/en/index)
  - [flash-attention](https://github.com/dao-ailab/flash-attention)
  - [Unsloth](https://unsloth.ai/)
- **Hardware:** 
  - Fine‑tuned on a single **[Tesla V100 (32GB VRAM)](https://www.nvidia.com/en-gb/data-center/tesla-v100/)**
  - Inference deployed on **[RTX 4090 (24GB VRAM)](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)**/**[A800 (80GB VRAM)](https://www.nvidia.com/en-us/products/workstations/a800/)**/**[RTX PRO 6000 (96GB VRAM)](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)**.
- **Data Split:** 
  - 75% Training
  - 15% Evaluation
  - 10% Test

👉[View the fine-tuned adapter here.](https://huggingface.co/lituokobe/Qwen3-4B-Instruct-2507-LoRA-Intent-Classifier)

## 📈 7. Results

### 7.1 Accuracy
| Model                                   | Test Set Accuracy     |
|:----------------------------------------|:----------------------|
| Qwen3-4B-Instruct-2507 (Baseline)       | 76.8%                 |
| **Qwen3-4B-Instruct-2507 (Fine‑Tuned)** | **93.5%**             |

The results clearly demonstrate the power of targeted fine‑tuning. The base model, while capable, are not specialized for this domain and achieved only ~76% accuracy.

When the fine-tuned model is deployed for the AI agent, I can also intuitively feel that the it clearly has a much better judgement.

### 7.2 Latency

| GPU Model    | Category                 | VRAM  | Latency    |
|:-------------|:-------------------------|:------|:-----------|
| **RTX 4090** | Consumer High-End        | 24GB  | **0.474s** |
| **A800**     | Data Center              | 80GB  | **0.393s** |
| **PRO 6000** | Professional Workstation | 96GB  | **0.370s** |

- **Real-Time Readiness:** Every hardware tier delivers desirable results, with inference times consistently below the 0.5s benchmark.
- **Consumer Accessibility:** The **RTX 4090** demonstrates impressive efficiency for a consumer card, performing only slightly slower than enterprise-grade silicon.
- **Workstation Dominance:** The **PRO 6000** leads the group with the fastest response time, benefiting from its massive 96GB VRAM and high memory bandwidth.
- **Enterprise Scaling:** While the **A800** is optimized for multi-tenant data center deployments, its full throughput potential is best realized in high-concurrency workloads rather than single-machine tests.

## ⚠️ 8. Limitations
- Performance depends on quality of synthetic data
- May degrade on unseen domains (e.g. conversation with non commercial event topics)
- Qwen3.5 reasoning mode introduces latency instability

## 🏁 9. Getting Started
*Please make sure you have an NVIDIA GPU with at least 32GB VRAM (for training) or 16GB VRAM (for inference)

### 9.1 Learn the training process

Download this repository, and set up the environment (Python 3.11 recommended).
```Bash
git clone https://github.com/lituokobe/Qwen3-Fine-Tuning.git

cd Qwen3-Fine-Tuning

pip install -r requirements
```
Download the base model and training data to their respective locations:
```Bash
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir models/Qwen/Qwen3-4B-Instruct-2507
  
huggingface-cli download --repo-type dataset lituokobe/LoRA-Samples-Intention-Classifier \
  --local-dir data/final
```

Explore the `LoRA_training.ipynb` file in following folders to understand the fine-tuning process:
- Qwen3___5-4B_unsloth
- Qwen3_fa
- Qwen3_reg
- Qwen3_unsloth

### 9.2 Deploy the model with LoRA adapter
If you would like to directly experience the fine-tuned model, download the adapter files:
```Bash
huggingface-cli download lituokobe/Qwen3-4B-Instruct-2507-LoRA-Intent-Classifier \
  --local-dir Qwen3_reg/lora_adapter_lr1e-4_1epoch
```

Deploy in the same environment:
``` shell
vllm serve models/Qwen/Qwen3-4B-Instruct-2507 \
    --enable-lora \
    --lora-modules qwen-finetune= Qwen3_reg/lora_adapter_lr1e-4_1epoch \
    --max-lora-rank 64 \
    --max-model-len 4096 \
    --trust-remote-code \
    --dtype float16 \
    --port 8000 \
    --served-model-name qwen3-4b
```

After the deployment is successful, you can test the model using a simple curl request:
``` shell
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "messages": [
      {
        "role": "user",
        "content": "Please take any sample from the training data or adapt it with the correct format"
      }
    ],
    "temperature": 0.0,
    "max_tokens": 50
  }'
```

The model is expected to return a structured dictionary (JSON) with the following schema:
``` JSON
{
  "input_summary": "A concise summary of the user's latest message.",
  "intention_id": "short_intent_label"
}
```
