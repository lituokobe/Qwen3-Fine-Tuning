# Qwen Fine-Tuning for Customer Service Intent Classification

<p align="center">
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
    <a href="https://github.com/your-username/your-repo-name/stargazers">
        <img src="https://img.shields.io/github/stars/your-username/your-repo-name?style=social" alt="GitHub stars">
    </a>
</p>

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Benefits](#-key-benefits)
- [Model Selection](#-model-selection)
- [Dataset Preparation](#-dataset-preparation)
    - [Data Generation Strategy](#data-generation-strategy)
    - [Quality Assurance](#quality-assurance)
- [Fine-Tuning Process](#-fine-tuning-process)
- [Results & Performance](#-results--performance)
- [Deployment](#-deployment)
- [Getting Started](#-getting-started)
- [Contributing](#-contributing)
- [License](#-license)

## 📌 Overview

My company is a **martech startup** that leverages technology to engage with clients at commercial events (car shows, renovation fairs, etc.).
Recently, we are pioneering using AI provide customer services via phone calls. This is the background of the project.

This repository provides a complete pipeline for fine-tuning small, efficient language models for **intention identification**. 
The fine-tuned model powers the Customer Service AI Agent Generator, enabling automated, intelligent, and highly controlled customer interactions.

By moving away from generic API-based LLMs, we deliver a solution that combines **specialized domain knowledge**, **low latency**, and **full data privacy**—all while being cost‑effective.

## 🚀 Key Benefits

Fine-tuning a small, local model offers a superior alternative to other LLM deployment strategies:

| Approach                                  | Cost                      | Response Time             | Control & Reliability        | Performance                 |
|:------------------------------------------|:--------------------------|:--------------------------|:-----------------------------|:----------------------------|
| **Commercial API Calls**                  | High (per-token cost)     | Slow (network latency)    | Low (dependent on 3rd party) | High                        |
| **Large Local Model (e.g., 70B)**         | Very High (hardware cost) | Fast (with high-end GPUs) | **Full**                     | **Very High**               |
| **Small Local Model (e.g., 4B)**          | Low                       | **Fast (<0.4s)**          | **Full**                     | **Moderate**                |
| **Fine-Tuned Small Model (This Project)** | **Low**                   | **Fast (<0.4s)**          | **Full**                     | **Production-Ready (98%+)** |

- **🏎️ Performance & Efficiency:** Deployed on a single RTX 3090/4090, inference time is under **0.4 seconds**, providing a seamless user experience.
- **🎯 Specialized Knowledge:** Trained on proprietary, domain-specific data, the model understands the nuances of your company's projects far better than a general-purpose model.
- **🔒 Full Control & Data Privacy:** Complete ownership of the model and infrastructure ensures reliable maintenance, easy debugging, and that all customer data remains in‑house.

## 🤖 Model Selection

We selected two state‑of‑the‑art, open‑source instruction‑tuned models from the Qwen family as our base models. Their small size and lack of built‑in reasoning make them ideal for a fast, focused classification task.

- **`Qwen3.5-4B`** – The flagship open‑sourced instruct small model from Qwen, with reasoning turned off. Perfect as the base model for fine‑tuning for intention identification of a chatbot.
- **`Qwen3-4B-Instruct-2507`** – The most popular open‑sourced instruct small model from Qwen, no reasoning process at all. Perfect as the base model for fine‑tuning and as a baseline comparison.

## 📊 Dataset Preparation

High‑quality, diverse data is the cornerstone of this project's success. The dataset was constructed through a multi‑stage process to ensure it covers the full spectrum of real‑world customer interactions.

### Data Generation Strategy
A total of **10,000+ training samples** were generated using a mixture of state‑of‑the‑art models. This blend ensures both high accuracy and diversity, including edge cases.

- **50% `GLM-4-Plus` (Zhipu)** – Used as the primary generator for its high accuracy and ability to produce diverse, in‑domain questions.
- **25% `GPT-4-Turbo` (OpenAI)** – Crucial for generating challenging edge cases, including rude, vulgar, or disrespectful language, which models with stricter safety filters often avoid.
- **15% `Kimi (Moonshot AI)`** – Adds further linguistic diversity. All outputs from this model undergo manual review for quality assurance.
- **5% `Gemini 1.5 Pro` (Google)** – Another non‑Chinese model used to add diversity, particularly for generating content related to culturally sensitive topics.
- **5% `Qwen-Max` (Alibaba)** – A high‑quality local model, though its outputs tend to follow predictable patterns, reducing overall dataset diversity.

### Quality Assurance
- **Initial Seed Data:** 100 manually crafted samples were created based on discussions with business managers to cover both common and niche scenarios.
- **Manual Review:** **20% of all generated data** was manually checked and corrected to ensure accuracy and relevance. The data covers a wide range of intentions: **positive, negative, neutral, rude, general inquiries, and highly specific, niche questions.**

## ⚙️ Fine-Tuning Process

- **Technique:** **LoRA (Low‑Rank Adaptation)** was chosen for its efficiency and modularity. It allows us to fine‑tune the model on a consumer‑grade GPU without catastrophic forgetting, and the resulting adapter can be easily merged or swapped.
- **Hardware:** Fine‑tuned on a single **NVIDIA RTX 4090 (24GB VRAM)** using FP16 mixed precision.
- **Data Split:** 75% Training | 15% Evaluation | 10% Test

## 📈 Results & Performance

The fine‑tuning process yielded exceptional results, transforming a generic small model into a production‑ready intent classifier.

| Model | Test Set Accuracy |
| :--- | :--- |
| Qwen3-4B-Instruct-2507 (Baseline) | 65% |
| **Qwen3-4B-Instruct-2507 (Fine‑Tuned)** | **98%** |
| Qwen3.5-4B (Baseline) | 68% |
| **Qwen3.5-4B (Fine‑Tuned)** | **99%** |

The results clearly demonstrate the power of targeted fine‑tuning. The base models, while capable, are not specialized for this domain and achieve only ~65‑68% accuracy. After fine‑tuning, both models achieve **over 98% accuracy**, making them highly reliable for production use in a customer service setting.

## 🚢 Deployment

```
git clone https://www.modelscope.cn/Qwen/Qwen3-4B-Instruct-2507.git
```

The **`Qwen3.5-4B` fine‑tuned model** has been selected for deployment due to its slightly higher performance. It is now powering the AI Agent Generator for the company's customer service applications, handling real‑world user intent classification with high speed and accuracy.

*See the [Getting Started](#-getting-started) section to load and use the fine‑tuned model.*

## 🏁 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- An NVIDIA GPU with at least 16GB VRAM (for training) or 8GB VRAM (for inference)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name