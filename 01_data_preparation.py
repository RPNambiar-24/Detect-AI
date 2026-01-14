import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def load_reddit_gpt_dataset(filepath='data/reddit_gpt_dataset.csv'):
    """Load Reddit GPT dataset (GRiD)"""
    print(f"Loading {filepath}...")
    
    df = pd.read_csv(filepath)
    print(f"Columns found: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    
    # Inspect first few rows to understand structure
    print(f"\nFirst few rows:\n{df.head()}")
    
    # Common column names in GRiD dataset:
    # 'text' or 'content' or 'body' for text
    # 'label' or 'class' or 'is_human' or 'generated' for labels
    
    # Try to identify text and label columns
    text_col = None
    label_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'text' in col_lower or 'content' in col_lower or 'body' in col_lower or 'post' in col_lower:
            text_col = col
        if 'label' in col_lower or 'class' in col_lower or 'human' in col_lower or 'generated' in col_lower or 'target' in col_lower:
            label_col = col
    
    if text_col and label_col:
        df_processed = pd.DataFrame({
            'text': df[text_col],
            'label': df[label_col]
        })
        
        # Ensure labels are 0 (human) and 1 (AI)
        unique_labels = df_processed['label'].unique()
        print(f"\nUnique labels: {unique_labels}")
        
        # Convert labels if needed
        if set(unique_labels) == {'human', 'ai'} or set(unique_labels) == {'human', 'gpt'}:
            df_processed['label'] = df_processed['label'].map({'human': 0, 'ai': 1, 'gpt': 1})
        elif set(unique_labels) == {True, False}:
            # If True=human, False=AI
            df_processed['label'] = (~df_processed['label']).astype(int)
        
        return df_processed
    
    return df

def load_article_level_data(filepath='data/article_level_data.csv'):
    """Load article-level dataset"""
    print(f"\nLoading {filepath}...")
    
    df = pd.read_csv(filepath)
    print(f"Columns found: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    
    # Identify text and label columns
    text_col = None
    label_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'text' in col_lower or 'article' in col_lower or 'content' in col_lower:
            text_col = col
        if 'label' in col_lower or 'class' in col_lower or 'generated' in col_lower:
            label_col = col
    
    if text_col and label_col:
        df_processed = pd.DataFrame({
            'text': df[text_col],
            'label': df[label_col]
        })
        return df_processed
    
    return df

def load_your_dataset(filepath='data/your_dataset_5000.csv'):
    """Load your prepared dataset"""
    print(f"\nLoading {filepath}...")
    
    df = pd.read_csv(filepath)
    print(f"Columns found: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    
    # This dataset might already be in the right format
    if 'text' in df.columns and 'label' in df.columns:
        return df[['text', 'label']]
    
    return df

def combine_and_prepare_datasets():
    """Combine all datasets and prepare final dataset"""
    
    all_dataframes = []
    
    # Load reddit_gpt_dataset (largest dataset)
    try:
        df_reddit = load_reddit_gpt_dataset('data/reddit_gpt_dataset.csv')
        if df_reddit is not None and 'text' in df_reddit.columns:
            all_dataframes.append(df_reddit)
            print(f"✓ Reddit GPT dataset loaded: {len(df_reddit)} samples")
    except Exception as e:
        print(f"✗ Error loading reddit_gpt_dataset.csv: {e}")
    
    # Load your_dataset_5000
    try:
        df_your = load_your_dataset('data/your_dataset_5000.csv')
        if df_your is not None and 'text' in df_your.columns:
            all_dataframes.append(df_your)
            print(f"✓ Your dataset loaded: {len(df_your)} samples")
    except Exception as e:
        print(f"✗ Error loading your_dataset_5000.csv: {e}")
    
    # Load article_level_data (optional)
    try:
        df_article = load_article_level_data('data/article_level_data.csv')
        if df_article is not None and 'text' in df_article.columns:
            all_dataframes.append(df_article)
            print(f"✓ Article-level dataset loaded: {len(df_article)} samples")
    except Exception as e:
        print(f"✗ Error loading article_level_data.csv: {e}")
    
    if not all_dataframes:
        raise ValueError("No datasets could be loaded! Please check your CSV files.")
    
    # Combine all datasets
    print("\n" + "="*60)
    print("Combining datasets...")
    df_combined = pd.concat(all_dataframes, ignore_index=True)
    
    # Clean data
    print("Cleaning data...")
    df_combined['text'] = df_combined['text'].astype(str).str.strip()
    df_combined = df_combined[df_combined['text'].str.len() > 20]  # Remove very short texts
    df_combined = df_combined.dropna(subset=['text', 'label'])
    df_combined = df_combined.drop_duplicates(subset=['text'])  # Remove duplicates
    
    # Ensure labels are integers
    df_combined['label'] = df_combined['label'].astype(int)
    
    print(f"\nCombined dataset shape: {df_combined.shape}")
    print(f"Label distribution before balancing:")
    print(df_combined['label'].value_counts())
    
    # Balance the dataset
    print("\nBalancing dataset...")
    min_class = df_combined['label'].value_counts().min()
    
    # Sample up to 10,000 from each class for manageable training
    sample_size = min(min_class, 10000)
    
    df_balanced = df_combined.groupby('label').sample(n=sample_size, random_state=42)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"\nFinal balanced dataset shape: {df_balanced.shape}")
    print(f"Final label distribution:")
    print(df_balanced['label'].value_counts())
    
    # Show sample texts
    print("\n" + "="*60)
    print("Sample Human Text:")
    print(df_balanced[df_balanced['label']==0].iloc[0]['text'][:200] + "...")
    print("\nSample AI Text:")
    print(df_balanced[df_balanced['label']==1].iloc[0]['text'][:200] + "...")
    print("="*60)
    
    # Save processed data
    df_balanced.to_csv('data/processed_data.csv', index=False)
    print(f"\n✓ Processed data saved to: data/processed_data.csv")
    
    return df_balanced

if __name__ == "__main__":
    print("="*60)
    print("DetectAI - Data Preparation")
    print("="*60 + "\n")
    
    df = combine_and_prepare_datasets()
    
    print("\n" + "="*60)
    print("✓ Data preparation complete!")
    print(f"✓ Total samples: {len(df)}")
    print(f"✓ Human samples: {(df['label']==0).sum()}")
    print(f"✓ AI samples: {(df['label']==1).sum()}")
    print("="*60)
    print("\nNext step: Run 'python 02_feature_extraction.py'")