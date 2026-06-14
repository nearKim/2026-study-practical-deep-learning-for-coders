# Natural Language Processing with Disaster Tweets

**Kaggle Competition:** [NLP Getting Started](https://www.kaggle.com/competitions/nlp-getting-started)

## Overview
This project involves building a machine learning model that predicts which Tweets are about real disasters and which ones are not. The dataset contains short and noisy tweets, making it an excellent beginner project for fine-tuning Hugging Face Transformers.

## Goals
1. **Binary Classification:** Predict if a tweet is a real disaster (`target=1`) or not (`target=0`).
2. **First Fine-tuning Practice:** Use `distilroberta-base` for binary sequence classification.
3. **Data Cleaning Impact:** Compare model performance before and after cleaning the noisy text (e.g., removing URLs, handles, and special characters).
4. **Validation Strategies:** Experiment with different validation splits to ensure the model generalizes well.

## How to Proceed
We will tackle this project step-by-step to solidify your deep learning and NLP knowledge:

1. **Step 1: Data Exploration & Preprocessing** (Loading data, basic cleaning).
2. **Step 2: Tokenization & Datasets** (Converting text to tensors using Hugging Face's Tokenizer and `Dataset` library).
3. **Step 3: Model Setup** (Initializing `distilroberta-base` for Sequence Classification).
4. **Step 4: Training Loop** (Using the Hugging Face `Trainer` API or writing a custom PyTorch training loop).
5. **Step 5: Evaluation & Iteration** (Testing data cleaning impact and validation splits).

You will write the core components of the code to practice what you've learned!
