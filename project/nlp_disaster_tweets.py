import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# ==============================================================================
# NLP with Disaster Tweets - Fine-Tuning distilroberta-base
# ==============================================================================
# Instructions:
# - Download the dataset from Kaggle: https://www.kaggle.com/competitions/nlp-getting-started/data
#   Place `train.csv` and `test.csv` in the `project/data/` folder.
# - Fill in the missing code where indicated by `# TODO` comments.

MODEL_NAME = "distilroberta-base"

# ==============================================================================
# STEP 1: Data Exploration & Preprocessing
# ==============================================================================
def load_and_clean_data(csv_path, clean_text=False):
    """
    Load the CSV data and optionally clean the tweets.
    """
    df = pd.read_csv(csv_path)
    
    if clean_text:
        # TODO: Implement basic text cleaning. 
        # (e.g., remove URLs, HTML tags, or @mentions using regex)
        # For now, just a placeholder function. You will experiment with this later!
        pass
        
    return df

# ==============================================================================
# STEP 2: Tokenization & Datasets
# ==============================================================================
def prepare_datasets(df, test_size=0.2, random_state=42):
    """
    Split the dataframe into train and validation sets, 
    and convert them into Hugging Face `Dataset` objects.
    """
    # TODO: Split `df` into `train_df` and `val_df` using train_test_split
    train_df, val_df = None, None
    
    # Convert pandas dataframes to Hugging Face Datasets
    # TODO: Create train_dataset and val_dataset
    train_dataset = None
    val_dataset = None
    
    return train_dataset, val_dataset

def tokenize_datasets(train_dataset, val_dataset, tokenizer):
    """
    Apply the tokenizer to the text column in the datasets.
    """
    def tokenize_function(examples):
        # TODO: Call the tokenizer on the 'text' column. 
        # Remember to set padding="max_length" and truncation=True
        return None
    
    # Apply tokenization using `.map()`
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    return tokenized_train, tokenized_val

# ==============================================================================
# STEP 3 & 4: Model Setup & Training Loop
# ==============================================================================
def compute_metrics(pred):
    """
    Compute accuracy and F1 score for the validation set.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # TODO: Calculate accuracy and f1 score using sklearn functions
    acc = None
    f1 = None
    
    return {"accuracy": acc, "f1": f1}

def train_model(tokenized_train, tokenized_val):
    """
    Initialize the model and train it using Hugging Face's Trainer API.
    """
    # TODO: Load the pre-trained model for Sequence Classification
    # Hint: We are doing binary classification (num_labels=2)
    model = None
    
    # TODO: Define TrainingArguments
    # Experiment with learning_rate, num_train_epochs, per_device_train_batch_size
    training_args = None
    
    # TODO: Initialize the Trainer
    trainer = None
    
    # Train the model
    print("Starting training...")
    # trainer.train()  # Uncomment this to run training when you are ready
    
    return trainer

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
if __name__ == "__main__":
    # 1. Load Data (assuming you have downloaded it to project/data/train.csv)
    # df = load_and_clean_data("project/data/train.csv", clean_text=False)
    
    # 2. Prepare Splits
    # train_ds, val_ds = prepare_datasets(df)
    
    # 3. Initialize Tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 4. Tokenize
    # tokenized_train, tokenized_val = tokenize_datasets(train_ds, val_ds, tokenizer)
    
    # 5. Train
    # trainer = train_model(tokenized_train, tokenized_val)
    
    # 6. Evaluate
    # metrics = trainer.evaluate()
    # print("Validation Metrics:", metrics)
    print("Code skeleton loaded! Start filling in the TODOs.")
