import pandas as pd
import numpy as np
import joblib
import lime
from lime.lime_text import LimeTextExplainer
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def explain_with_lime():
    """Generate LIME explanations"""
    
    # Load models
    model = joblib.load('models/tfidf_logreg.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    
    df = pd.read_csv('data/processed_data.csv')
    
    # Prediction function for LIME
    def predict_proba(texts):
        return model.predict_proba(vectorizer.transform(texts))
    
    # Initialize LIME explainer
    explainer = LimeTextExplainer(class_names=['Human', 'AI'])
    
    # Select test instances
    test_samples = df.sample(5, random_state=42)
    
    print("Generating LIME Explanations...\n")
    print("="*60)
    
    for idx, row in test_samples.iterrows():
        text = row['text']
        true_label = 'Human' if row['label'] == 0 else 'AI'
        
        # Generate explanation
        exp = explainer.explain_instance(text, predict_proba, num_features=10)
        
        print(f"\nText {idx}: {text[:100]}...")
        print(f"True Label: {true_label}")
        pred_proba = predict_proba([text])[0]
        predicted = 'AI' if pred_proba[1] > 0.5 else 'Human'
        print(f"Predicted: {predicted} (AI: {pred_proba[1]:.2%}, Human: {pred_proba[0]:.2%})")
        print("\nTop contributing words:")
        for word, weight in exp.as_list():
            direction = "→ AI" if weight > 0 else "→ Human"
            print(f"  '{word}': {weight:+.4f} {direction}")
        
        # Save visualization
        try:
            fig = exp.as_pyplot_figure()
            plt.tight_layout()
            plt.savefig(f'results/lime_explanation_{idx}.png', dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  Warning: Could not save plot - {e}")
        
        print("-"*60)
    
    print("\n✓ LIME explanations saved to results folder!")

def explain_with_shap():
    """Generate SHAP explanations"""
    
    # Load models
    model = joblib.load('models/random_forest.pkl')
    df = pd.read_csv('data/features_data.csv')
    
    feature_cols = [col for col in df.columns if col not in ['text', 'label']]
    X = df[feature_cols].sample(min(200, len(df)), random_state=42)
    
    print("\nGenerating SHAP Explanations...")
    print("="*60)
    print(f"Analyzing {len(X)} samples with {len(feature_cols)} features...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    print(f"SHAP values shape: {np.array(shap_values).shape}")
    print(f"Feature matrix shape: {X.shape}")
    
    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        # For binary classification, use class 1 (AI-generated)
        shap_values_to_plot = shap_values[1]
    else:
        # Single array output
        shap_values_to_plot = shap_values
    
    # If still 3D (samples, features, classes), select class 1
    if len(shap_values_to_plot.shape) == 3:
        shap_values_to_plot = shap_values_to_plot[:, :, 1]
    
    print(f"Processed SHAP values shape: {shap_values_to_plot.shape}")
    
    # Summary plot (beeswarm)
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_to_plot, X, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig('results/shap_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ SHAP summary plot saved!")
    except Exception as e:
        print(f"Warning: Could not create summary plot - {e}")
    
    # Bar plot of mean absolute SHAP values
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_to_plot, X, plot_type="bar", show=False, max_display=20)
        plt.tight_layout()
        plt.savefig('results/shap_bar.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ SHAP bar plot saved!")
    except Exception as e:
        print(f"Warning: Could not create bar plot - {e}")
    
    # Feature importance (ensure 1D array)
    mean_abs_shap = np.abs(shap_values_to_plot).mean(axis=0)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*60)
    print("Top 15 Most Important Features (by SHAP):")
    print("="*60)
    for idx, row in feature_importance.head(15).iterrows():
        print(f"{row['feature']:.<40} {row['importance']:.6f}")
    print("="*60)
    
    feature_importance.to_csv('results/shap_importance.csv', index=False)
    print("\n✓ SHAP feature importance saved to: results/shap_importance.csv")
    
    # Waterfall plot for a single prediction
    try:
        # Get expected value
        if isinstance(explainer.expected_value, list):
            expected_val = explainer.expected_value[1]
        elif isinstance(explainer.expected_value, np.ndarray):
            expected_val = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
        else:
            expected_val = explainer.expected_value
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap.Explanation(
            values=shap_values_to_plot[0],
            base_values=expected_val,
            data=X.iloc[0].values,
            feature_names=feature_cols
        ), max_display=15, show=False)
        plt.tight_layout()
        plt.savefig('results/shap_waterfall.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ SHAP waterfall plot saved!")
    except Exception as e:
        print(f"Warning: Could not create waterfall plot - {e}")
    
    # Force plot for single prediction (HTML)
    try:
        if isinstance(explainer.expected_value, list):
            expected_val = explainer.expected_value[1]
        elif isinstance(explainer.expected_value, np.ndarray):
            expected_val = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
        else:
            expected_val = explainer.expected_value
            
        shap.force_plot(
            expected_val,
            shap_values_to_plot[0],
            X.iloc[0],
            matplotlib=True,
            show=False
        )
        plt.savefig('results/shap_force.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ SHAP force plot saved!")
    except Exception as e:
        print(f"Warning: Could not create force plot - {e}")

def analyze_model_insights():
    """Generate additional insights from trained models"""
    
    print("\n" + "="*60)
    print("Analyzing Model Insights...")
    print("="*60)
    
    # Load feature importance from Random Forest
    try:
        rf_importance = pd.read_csv('results/feature_importance.csv')
        shap_importance = pd.read_csv('results/shap_importance.csv')
        
        # Compare Random Forest and SHAP feature importance
        comparison = pd.merge(
            rf_importance.head(20)[['feature', 'importance']].rename(columns={'importance': 'rf_importance'}),
            shap_importance.head(20)[['feature', 'importance']].rename(columns={'importance': 'shap_importance'}),
            on='feature',
            how='outer'
        ).fillna(0)
        
        comparison = comparison.sort_values('rf_importance', ascending=False)
        
        print("\nTop Features Comparison (Random Forest vs SHAP):")
        print("="*60)
        print(f"{'Feature':<30} {'RF Importance':>15} {'SHAP Importance':>15}")
        print("-"*60)
        for _, row in comparison.head(15).iterrows():
            print(f"{row['feature']:<30} {row['rf_importance']:>15.6f} {row['shap_importance']:>15.6f}")
        print("="*60)
        
        comparison.to_csv('results/feature_comparison.csv', index=False)
        print("\n✓ Feature comparison saved to: results/feature_comparison.csv")
        
    except Exception as e:
        print(f"Could not generate comparison: {e}")
    
    # Generate summary report
    print("\n" + "="*60)
    print("Key Insights for AI Text Detection:")
    print("="*60)
    
    try:
        top_features = pd.read_csv('results/shap_importance.csv').head(10)
        
        print("\nMost discriminative features:")
        for idx, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"{idx}. {row['feature']}")
        
        print("\nThese features help distinguish AI-generated from human text.")
        print("Typically, AI text shows:")
        print("  - More consistent sentence/word lengths")
        print("  - Higher readability scores")
        print("  - More function words and connectors")
        print("  - Less lexical diversity")
        print("  - Fewer personal pronouns and emotions")
        
    except Exception as e:
        print(f"Could not generate insights: {e}")

if __name__ == "__main__":
    print("="*60)
    print("DetectAI - Explainability Analysis")
    print("="*60 + "\n")
    
    # Run LIME explanations
    explain_with_lime()
    
    # Run SHAP explanations
    explain_with_shap()
    
    # Generate insights
    analyze_model_insights()
    
    print("\n" + "="*60)
    print("✓ Explainability analysis complete!")
    print("✓ Check the 'results' folder for visualizations:")
    print("  - lime_explanation_*.png")
    print("  - shap_summary.png")
    print("  - shap_bar.png")
    print("  - shap_waterfall.png")
    print("  - shap_force.png")
    print("  - feature_comparison.csv")
    print("="*60)
    print("\nNext step: Run 'python 07_inference.py'")
