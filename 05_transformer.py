import pandas as pd
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def prepare_dataset():
    """Prepare dataset for transformer training"""
    
    df = pd.read_csv('data/processed_data.csv')
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
    
    return train_dataset, test_dataset

def tokenize_function(examples, tokenizer):
    """Tokenize texts"""
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

def compute_metrics(pred):
    """Compute evaluation metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_roberta():
    """Train RoBERTa model"""
    
    print("Loading RoBERTa model...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    
    # Prepare datasets
    train_dataset, test_dataset = prepare_dataset()
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Tokenize
    print("\nTokenizing datasets...")
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    # Create output directory
    os.makedirs('./results/roberta', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # Training arguments (FIXED PARAMETERS)
    training_args = TrainingArguments(
        output_dir='./results/roberta',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy='epoch',  # CHANGED from evaluation_strategy
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to='none'  # Disable wandb/tensorboard logging
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train
    print("\n" + "="*60)
    print("Training RoBERTa...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluating model...")
    print("="*60 + "\n")
    
    results = trainer.evaluate()
    
    print("\nRoBERTa Results:")
    print("="*60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("="*60)
    
    # Save model
    print("\nSaving model...")
    os.makedirs('models/roberta_model', exist_ok=True)
    model.save_pretrained('models/roberta_model')
    tokenizer.save_pretrained('models/roberta_model')
    
    print("\n" + "="*60)
    print("✓ RoBERTa training complete!")
    print("✓ Model saved to: models/roberta_model")
    print("="*60)
    
    return model, tokenizer

if __name__ == "__main__":
    print("="*60)
    print("DetectAI - RoBERTa Transformer Training")
    print("="*60 + "\n")
    
    model, tokenizer = train_roberta()
    
    print("\nNext step: Run 'python 06_explainability.py'")