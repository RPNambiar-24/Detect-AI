import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import spacy
import textstat
from collections import Counter
import re

nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

class StyleometricFeatureExtractor:
    
    def extract_features(self, text):
        """Extract comprehensive stylometric and linguistic features"""
        
        features = {}
        
        # Basic text statistics
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        features['word_count'] = len(words)
        features['char_count'] = len(text)
        features['sentence_count'] = len(sentences)
        
        # Lexical diversity
        unique_words = set(words)
        features['unique_word_count'] = len(unique_words)
        features['type_token_ratio'] = len(unique_words) / len(words) if len(words) > 0 else 0
        
        # Hapax legomenon (words appearing once)
        word_freq = Counter(words)
        hapax = sum(1 for count in word_freq.values() if count == 1)
        features['hapax_legomenon_ratio'] = hapax / len(words) if len(words) > 0 else 0
        
        # Average lengths
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
        
        # Readability metrics
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        features['gunning_fog'] = textstat.gunning_fog(text)
        features['smog_index'] = textstat.smog_index(text)
        features['coleman_liau_index'] = textstat.coleman_liau_index(text)
        
        # Function words and stop words
        function_words = [w for w in words if w in stop_words]
        features['function_word_ratio'] = len(function_words) / len(words) if words else 0
        
        # POS tags using spaCy
        doc = nlp(text)
        pos_counts = Counter([token.pos_ for token in doc])
        total_pos = sum(pos_counts.values())
        
        features['noun_ratio'] = pos_counts.get('NOUN', 0) / total_pos if total_pos > 0 else 0
        features['verb_ratio'] = pos_counts.get('VERB', 0) / total_pos if total_pos > 0 else 0
        features['adj_ratio'] = pos_counts.get('ADJ', 0) / total_pos if total_pos > 0 else 0
        features['adv_ratio'] = pos_counts.get('ADV', 0) / total_pos if total_pos > 0 else 0
        features['adp_ratio'] = pos_counts.get('ADP', 0) / total_pos if total_pos > 0 else 0
        features['aux_ratio'] = pos_counts.get('AUX', 0) / total_pos if total_pos > 0 else 0
        
        # Named entities
        entities = [ent.label_ for ent in doc.ents]
        features['entity_count'] = len(entities)
        features['entity_density'] = len(entities) / len(words) if words else 0
        
        # Punctuation analysis
        features['comma_count'] = text.count(',')
        features['semicolon_count'] = text.count(';')
        features['colon_count'] = text.count(':')
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['punctuation_density'] = sum([features['comma_count'], 
                                                features['semicolon_count'],
                                                features['exclamation_count'],
                                                features['question_count']]) / len(words) if words else 0
        
        # Discourse markers
        discourse_markers = ['however', 'moreover', 'furthermore', 'therefore', 'thus', 'hence', 'consequently']
        features['discourse_marker_count'] = sum(text.lower().count(marker) for marker in discourse_markers)
        
        # Negation words
        negation_words = ['not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'nobody']
        features['negation_count'] = sum(text.lower().count(neg) for neg in negation_words)
        
        return features
    
    def extract_batch(self, texts):
        """Extract features for multiple texts"""
        features_list = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"Processing text {i}/{len(texts)}")
            features_list.append(self.extract_features(text))
        return pd.DataFrame(features_list)

def create_feature_dataset():
    """Load data and extract features"""
    
    df = pd.read_csv('data/processed_data.csv')
    
    extractor = StyleometricFeatureExtractor()
    features_df = extractor.extract_batch(df['text'].values)
    
    # Combine with original data
    final_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    
    # Save features
    final_df.to_csv('data/features_data.csv', index=False)
    
    print(f"\nFeature extraction complete!")
    print(f"Total features: {len(features_df.columns)}")
    print(f"\nFeature names: {list(features_df.columns)}")
    
    return final_df

if __name__ == "__main__":
    df = create_feature_dataset()
