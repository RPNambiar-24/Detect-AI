import pandas as pd
import numpy as np
import joblib
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
import sys
import os

# Import feature extractor directly from file
import importlib.util
spec = importlib.util.spec_from_file_location("feature_extraction", "02_feature_extraction.py")
feature_extraction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_extraction)
StyleometricFeatureExtractor = feature_extraction.StyleometricFeatureExtractor

class DetectAIEnsemble:
    def __init__(self):
        print("Loading models...")
        
        # Load traditional ML models
        self.tfidf_model = joblib.load('models/tfidf_logreg.pkl')
        self.tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        self.rf_model = joblib.load('models/random_forest.pkl')
        
        # Load feature extractor
        self.feature_extractor = StyleometricFeatureExtractor()
        
        # Try to load transformer model (RoBERTa or DistilBERT)
        self.has_transformer = False
        self.transformer_type = None
        
        # Try RoBERTa first
        if os.path.exists('models/roberta_model'):
            try:
                print("Loading RoBERTa model...")
                self.transformer_tokenizer = RobertaTokenizer.from_pretrained('models/roberta_model')
                self.transformer_model = RobertaForSequenceClassification.from_pretrained('models/roberta_model')
                self.transformer_model.eval()
                self.has_transformer = True
                self.transformer_type = 'roberta'
                print("✓ RoBERTa loaded")
            except Exception as e:
                print(f"Could not load RoBERTa: {e}")
        
        # Try DistilBERT if RoBERTa not found
        if not self.has_transformer and os.path.exists('models/distilbert_model'):
            try:
                print("Loading DistilBERT model...")
                self.transformer_tokenizer = DistilBertTokenizer.from_pretrained('models/distilbert_model')
                self.transformer_model = DistilBertForSequenceClassification.from_pretrained('models/distilbert_model')
                self.transformer_model.eval()
                self.has_transformer = True
                self.transformer_type = 'distilbert'
                print("✓ DistilBERT loaded")
            except Exception as e:
                print(f"Could not load DistilBERT: {e}")
        
        if not self.has_transformer:
            print("⚠ No transformer model found, using baseline models only")
        
        print("✓ All available models loaded successfully!\n")
    
    def predict(self, text):
        """Predict using ensemble of models"""
        
        predictions = []
        confidences = []
        model_names = []
        
        # TF-IDF + Logistic Regression
        try:
            tfidf_vec = self.tfidf_vectorizer.transform([text])
            tfidf_pred = self.tfidf_model.predict_proba(tfidf_vec)[0]
            predictions.append(tfidf_pred[1])
            confidences.append(max(tfidf_pred))
            model_names.append('TF-IDF + LogReg')
        except Exception as e:
            print(f"TF-IDF prediction failed: {e}")
        
        # Random Forest on features
        try:
            features = self.feature_extractor.extract_features(text)
            features_df = pd.DataFrame([features])
            rf_pred = self.rf_model.predict_proba(features_df)[0]
            predictions.append(rf_pred[1])
            confidences.append(max(rf_pred))
            model_names.append('Random Forest')
        except Exception as e:
            print(f"Random Forest prediction failed: {e}")
        
        # Transformer (RoBERTa or DistilBERT)
        if self.has_transformer:
            try:
                inputs = self.transformer_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.transformer_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)[0]
                    predictions.append(probs[1].item())
                    confidences.append(max(probs).item())
                    model_names.append(self.transformer_type.upper())
            except Exception as e:
                print(f"Transformer prediction failed: {e}")
        
        # Ensemble prediction (weighted average)
        if self.has_transformer:
            weights = [0.2, 0.2, 0.6]  # Give more weight to transformer
        else:
            weights = [0.5, 0.5]
        
        ensemble_score = sum(p * w for p, w in zip(predictions, weights))
        final_label = 'AI-Generated' if ensemble_score > 0.5 else 'Human-Written'
        confidence = max(ensemble_score, 1 - ensemble_score)
        
        result = {
            'label': final_label,
            'confidence': confidence,
            'ai_probability': ensemble_score,
            'human_probability': 1 - ensemble_score,
            'individual_predictions': {}
        }
        
        # Add individual model predictions
        for name, pred in zip(model_names, predictions):
            result['individual_predictions'][name] = pred
        
        return result

def test_inference():
    """Test the inference pipeline"""
    
    ensemble = DetectAIEnsemble()
    
    test_texts = [
        "I really enjoyed the movie last night! The plot was engaging and the acting was superb. Can't wait to see what they do next.",
        "The implementation of the aforementioned methodology demonstrates significant improvements in performance metrics across multiple domains.",
        "Hey! What's up? I'm so excited about the weekend plans. Can't wait to see you! Let me know if you're free.",
        "In conclusion, the systematic analysis presented herein provides comprehensive insights into the subject matter under consideration.",
        "My dog does the funniest thing when I come home - he runs in circles for like 5 minutes straight lol. It's adorable but also kinda weird.",
        "Furthermore, it is important to note that the integration of these components contributes to enhanced functionality and effectiveness.",
        "Just finished organizing my closet and found clothes I completely forgot about! Some of them still have tags on them haha.",
        "The utilization of advanced techniques facilitates the achievement of desired outcomes in an efficient and systematic manner."
    ]
    
    print("="*80)
    print("DetectAI Inference Results")
    print("="*80 + "\n")
    
    for i, text in enumerate(test_texts, 1):
        result = ensemble.predict(text)
        
        print(f"\n{'='*80}")
        print(f"Text {i}:")
        print(f"{text}")
        print(f"\n{'-'*80}")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"AI Probability: {result['ai_probability']:.2%}")
        print(f"Human Probability: {result['human_probability']:.2%}")
        print(f"\nIndividual Model Predictions:")
        for model_name, prob in result['individual_predictions'].items():
            print(f"  {model_name:.<30} {prob:.2%} (AI)")
        print("="*80)
    
    print("\n✓ Inference testing complete!")

def predict_from_user_input():
    """Interactive prediction mode"""
    
    ensemble = DetectAIEnsemble()
    
    print("\n" + "="*80)
    print("DetectAI - Interactive Mode")
    print("="*80)
    print("Enter text to classify (or 'quit' to exit)\n")
    
    while True:
        print("\nEnter text: ", end='')
        text = input().strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if len(text) < 10:
            print("⚠ Please enter at least 10 characters.")
            continue
        
        result = ensemble.predict(text)
        
        print("\n" + "-"*80)
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"AI Probability: {result['ai_probability']:.2%}")
        print(f"Human Probability: {result['human_probability']:.2%}")
        print("\nModel Breakdown:")
        for model_name, prob in result['individual_predictions'].items():
            print(f"  {model_name}: {prob:.2%} (AI)")
        print("-"*80)

if __name__ == "__main__":
    print("="*80)
    print("DetectAI - Inference Pipeline")
    print("="*80 + "\n")
    
    print("Choose mode:")
    print("1. Run test with sample texts")
    print("2. Interactive mode (enter your own text)")
    print("\nEnter choice (1 or 2): ", end='')
    
    try:
        choice = input().strip()
        
        if choice == '1':
            test_inference()
        elif choice == '2':
            predict_from_user_input()
        else:
            print("Invalid choice. Running test mode...")
            test_inference()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    
    print("\nNext step: Run 'python 08_gradio_app.py' for web interface")
