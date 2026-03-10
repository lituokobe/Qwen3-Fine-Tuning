
This project aims to fine-tune a small model which can be deployed locally and reason efficiently to suit the intention identification
purpose for the project of Customer Service AI Agent Generator.

Meanwhile, fine-tuning can highly customize the model to make it specialized for intention identification for customer service in the 
marketing projects from my company's industry, including car shows, renovation fairs, etc. Highly improves the level of control.

Compared to use API calls that incurs on-demand cost and long response time or local deployment of large model that cost a lot, or local deployment 
of small model that doesn't perform too well, using a fine-tuned small model can
- deliver the great balance between cost and efficiently (locally deployed on RTX 4090 or 3090, response time no more than 0.4s)
- specialized knowledge to serve the company's projects, with dedicated training data prepared for the training
- reliable maintenance and debugging as the whole LLM service is domestically owned

## Models selected:
Qwen3-4B-Instruct-2507, the most popular open-sourced instruct small model from Qwen, no reasoning process at all. Perfect as the base model
for the fine-tuning for intention identification of chatbot.

Qwen3.5-4B, the flagship open-sourced instruct small model from Qwen, reasoning turned off. Perfect as the base model
for the fine-tuning for intention identification of chatbot.

## Data preparation
Two stages of data preparation
- generation of 100 samples with 2 built workflows
- Discussion with business managers for common recipient questions, together with niche ones.
- Understand the business scope, use API calls from popular flaship models to generate another 10K samples as training data. I majorly used models from China as the data needs to be in Chinses:
  - 50%: GLM-4.7, flagship language model from Zhipu, deliver both diversity and accuracy, key data generating model
  - 25%: GPT-4.1, most advance non-reasoning model from OpenAI - use it to generate rude/vulgar/disrespecting recipient intentions which is limited by models from China
  - 15%: Kimi-2.5, flagship model from Moonshot, providing diversity, but not as accurate as GLM-4.7, need manual check
  - 5%: Gemini, another model not from China, adding diversity to generate sensitive Chinese content
  - 5%: Qwen-max, flagship model from Alibaba, high quality generation, but easy to follow same pattern

20% of the generated data is manually checked, and the generation covers different topic directions: e.g. positive, negative, rude, asking common questions, niche questions.

## Fine-tuning techniques
LoRA - most popular fine-tuning techniques for its high performance and modularization
Fine-tuned with FP16 with Nvidia GTX 4090 - using full precision on a the best class individual level GPU
75% - training set, 15% evaluation set, 10% test set

## Result:
With this fine-tuning practice, on the performance of test set:
Qwen3-4B-Instruct-2507: 65%
Qwen3-4b-Instruct-2507-fine-tuned: 98%
Qwen3.5-4B: 68%
Qwen3.5-4B-fine-tuned: 99%

This shows that the fine-tuning practice really improves the model to production ready level and serve the customer service purposes for the company's dedicated business.
Qwen3.5-4B-fine-tuned is deployed for the company's AI agent generator already.