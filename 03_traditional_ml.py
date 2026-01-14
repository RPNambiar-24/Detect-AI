import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_tfidf_model():
    """Train TF-IDF + Logistic Regression baseline"""
    
    df = pd.read_csv('data/processed_data.csv')
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluation
    print("TF-IDF + Logistic Regression Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    
    # Save model
    joblib.dump(model, 'models/tfidf_logreg.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    
    return model, vectorizer

def train_random_forest():
    """Train Random Forest on stylometric features"""
    
    df = pd.read_csv('data/features_data.csv')
    
    # Select feature columns (all except text and label)
    feature_cols = [col for col in df.columns if col not in ['text', 'label']]
    X = df[feature_cols]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    print("\n\nRandom Forest (Stylometric Features) Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save feature importance
    feature_importance.to_csv('results/feature_importance.csv', index=False)
    
    # Save model
    joblib.dump(model, 'models/random_forest.pkl')
    
    return model, feature_importance

if __name__ == "__main__":
    print("Training Traditional ML Models...\n")
    tfidf_model, vectorizer = train_tfidf_model()
    rf_model, feat_imp = train_random_forest()
    print("\nTraining complete!")
