# [Try Next-Word Predictor Sandbox](https://zackakil.github.io/Next-Word-Predictor-Sandbox/)

A simple, interactive web app designed to demonstrate how Large Language Models (LLMs) predict text one word/token at a time based on training data probabilities.

## How It Works

1. **Training Data**: Input example sentences. The model learns frequency counts and probabilities for tokens following each word context.
2. **Learned Probability Table**: Inspect the learned probabilities for every context.
3. **Model Inference**: Provide a starting prompt, select a token sampling strategy (*Probability-Weighted*, *Greedy*, or *Pure Random*), and observe step-by-step token generation.
