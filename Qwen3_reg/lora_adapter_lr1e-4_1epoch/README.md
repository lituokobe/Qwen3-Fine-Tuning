---
license: apache-2.0
language:
- zh
- en
metrics:
- accuracy
base_model:
- Qwen/Qwen3-4B-Instruct-2507
pipeline_tag: text-classification
tags:
- agent
- commercial
- customer-engagement
---

# Model Card for Qwen3-4B-Instruct-2507-LoRA-Intent-Classifier
Identify user intention in multi-turn conversations for AI-driven client engagement.


## Model Details

### Model Description
This repository contains a LoRA (Low-Rank Adaptation) adapter fine-tuned on top of the base model [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)for user intent classification.

The model is designed to analyze the latest user utterance within a conversation and classify it into a predefined set of intention categories. These categories are customizable and defined by business users (e.g., sales or customer service managers).

The adapter is lightweight and intended to be merged with the base model at inference time.


- **Developed by:** [Li Tuo](https://github.com/lituokobe)
- **Model type:** Causal Language Model (LLM) with LoRA adapter for classification
- **Language(s) (NLP):** Chinese (primary), English (partial support)
- **License:** Apache-2.0
- **Finetuned from model:** [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

## Uses

This model is intended to be used as a self-hosted intent classification module within AI-powered customer engagement systems, such as:

- AI marketing agents 
- Customer service chatbots 
- Phone call automation systems

It is a core component of the project:
👉 [AI CS agent generator](https://github.com/lituokobe/AI-CS-Agent)

### Direct Use

Given a prompt that includes customized intention classes and a multi-turn conversation, the model:

- Focuses on the latest user input 
- Extracts a structured intent label 
- Outputs a predefined intent ID

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Use the code below to get started with the model.

[More Information Needed]

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

[More Information Needed]

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

[More Information Needed]


#### Training Hyperparameters

- **Training regime:** [More Information Needed] <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

[More Information Needed]

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

[More Information Needed]

### Results

[More Information Needed]

#### Summary



## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

[More Information Needed]

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]
### Framework versions

- PEFT 0.18.1