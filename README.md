# 🔍 DetectAI: Human vs AI-Generated Text Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/Gradio-Interface-orange)](https://gradio.app/)

An advanced machine learning system designed to distinguish between human-written and AI-generated text using a hybrid approach combining stylometric analysis, traditional machine learning, deep learning, and transformer-based models with explainable AI.

## 📖 Overview

DetectAI addresses the growing challenge of identifying AI-generated content in the era of sophisticated language models like GPT-4, Claude, and Gemini. The system employs multiple complementary detection strategies to achieve high accuracy while maintaining interpretability through explainable AI techniques.

## 🎯 What It Does

DetectAI analyzes text input and provides:
- **Binary classification**: Human-written or AI-generated
- **Confidence scores**: Probability distribution for each class
- **Model-level breakdown**: Individual predictions from TF-IDF, Random Forest, BiLSTM, and Transformer models
- **Explainability**: Visual explanations showing which features and words influenced the decision

## 📊 Performance Results

Trained and evaluated on 1,472 text samples from diverse sources (Reddit posts, articles, academic texts):

### Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| TF-IDF + Logistic Regression | 87.46% | 0.87 | 0.88 | 0.87 |
| Random Forest (Stylometric) | 91.19% | 0.93 | 0.89 | 0.91 |
| BiLSTM with Attention | 88.81% | 0.94 | 0.83 | 0.88 |
| **RoBERTa (Fine-tuned)** | **98.64%** | **0.98** | **0.99** | **0.99** |
| **Ensemble System** | **~98-99%** | **~0.98** | **~0.99** | **~0.98** |

### Key Performance Highlights

✨ **98.64% accuracy** with RoBERTa transformer - exceeding published benchmarks.

✨ **99.32% recall** - catches 99.32% of all AI-generated texts

✨ **97.99% precision** - when flagging text as AI, correct 98% of the time

✨ **91.19% accuracy** using only stylometric features - demonstrating strong interpretable baseline.
### Top 10 Most Discriminative Features

Identified through Random Forest feature importance analysis :

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | Type-Token Ratio | 0.089 | Lexical diversity measure |
| 2 | Hapax Legomenon Ratio | 0.086 | Words appearing only once |
| 3 | Noun Ratio | 0.085 | Proportion of nouns (POS) |
| 4 | Flesch-Kincaid Grade | 0.058 | Readability score |
| 5 | Average Word Length | 0.055 | Word complexity metric |
| 6 | Character Count | 0.053 | Text length |
| 7 | Coleman-Liau Index | 0.050 | Readability metric |
| 8 | Gunning Fog Index | 0.049 | Readability complexity |
| 9 | Unique Word Count | 0.038 | Vocabulary richness |
| 10 | Function Word Ratio | 0.038 | Stop word density |

### Comparison to Published Research

DetectAI matches or exceeds state-of-the-art results :

| Study/Benchmark | Accuracy | DetectAI Result |
|-----------------|----------|-----------------|
| Nature study (2025) - RoBERTa  | 96.1% | **98.64%** ✅ |
| BiLSTM baseline | ~88-89% | 88.81% ✅ |
| Traditional ML methods | ~80-85% | 87-91% ✅ |

## 🔬 How It Works

### Architecture

DetectAI uses a **hybrid ensemble approach** combining four complementary detection methods:

**1. Stylometric Feature Analysis**
- Extracts 30+ quantitative features measuring writing style patterns:
  - **Lexical diversity**: Type-Token Ratio, hapax legomenon (words appearing once), unique word counts
  - **Syntactic complexity**: Sentence length distribution, POS tag ratios (nouns, verbs, adjectives, adverbs, prepositions, auxiliary verbs)
  - **Readability metrics**: Flesch Reading Ease, Gunning Fog Index, SMOG Index, Coleman-Liau Index
  - **Function word patterns**: Stop word density, discourse markers ("however," "moreover," "therefore"), negation frequency
  - **Punctuation usage**: Comma, semicolon, exclamation mark, and question mark densities
  - **Named entity statistics**: Entity counts and density per text

AI-generated text typically shows more consistent sentence lengths, higher readability scores, increased function word usage, and lower lexical diversity compared to human writing.

**2. Traditional Machine Learning**
- **TF-IDF + Logistic Regression**: Captures n-gram patterns (word sequences) that distinguish AI from human text, achieving 87.46% accuracy
- **Random Forest**: Trained on 30 stylometric features to identify complex non-linear relationships between writing style metrics, achieving 91.19% accuracy with high interpretability.

**3. Deep Learning**
- **BiLSTM with Attention**: Bidirectional Long Short-Term Memory network that captures sequential patterns and long-range dependencies in text, achieving 88.81% accuracy with 94% precision.

**4. Transformer-Based Deep Learning**
- **Fine-tuned RoBERTa**: Pre-trained language model adapted specifically for human vs AI text classification.
- Learns contextual representations capturing subtle linguistic patterns that traditional methods miss
- Achieves 98.64% accuracy with 99.32% recall through deep semantic understanding.

**5. Ensemble Decision**
- Combines predictions from all models using weighted voting (60% RoBERTa, 20% TF-IDF, 20% Random Forest).
- Final prediction represents consensus across complementary detection approaches
- Confidence score reflects agreement between models
- Expected ensemble accuracy: **~98-99%**

### Explainability Layer

DetectAI integrates two complementary explainability techniques:

**LIME (Local Interpretable Model-Agnostic Explanations)**
- Highlights specific words contributing to each prediction.
- Shows whether words push the classification toward "human" or "AI"
- Provides instance-level explanations for individual texts
- Example findings: Words like "and," "data," "models" indicate AI; "might," "but," "there" indicate human

**SHAP (SHapley Additive exPlanations)**
- Calculates global feature importance across all predictions.
- Identifies which stylometric features (readability, function words, etc.) are most discriminative
- Based on game theory principles for consistent, reliable explanations
- Confirmed top features: type-token ratio (0.059), noun ratio (0.056), hapax legomenon ratio (0.056)

### Key Detection Signals

Research shows AI-generated text exhibits characteristic patterns [web:1][web:5][web:10]:

✅ **More common in AI text:**
- Higher frequency of function words and connectors ("and," "are," "of," "to," "without")
- More uniform sentence structure and length
- Elevated readability scores (easier to read)
- Consistent use of formal discourse markers ("furthermore," "moreover," "in conclusion")
- Lower hapax legomenon ratio (fewer unique words used once)
- Lower type-token ratio (less lexical diversity)

✅ **More common in human text:**
- Greater lexical diversity and vocabulary richness
- Variable sentence complexity
- More personal pronouns and emotional language
- Irregular punctuation patterns
- Higher entity density in conversational contexts
- More hedging language ("might," "could," "perhaps")

## 📈 Training Data

The system is trained on diverse datasets totaling 1,472 balanced samples [web:11][web:24][web:18]:

- **Reddit posts** (human) vs GPT-3.5/GPT-4 generations (GRiD dataset) covering casual conversation
- **News articles** from authentic sources vs synthetic versions from multiple LLMs
- **Academic and formal writing** samples to capture diverse domains

This multi-domain training ensures robustness across different text types and writing styles.

## 🎨 Web Interface

Built with Gradio, the interface provides:
- Text input area for classification requests
- Real-time prediction with visual confidence indicators
- Probability distribution chart (Human vs AI percentages)
- Individual model predictions breakdown (TF-IDF, Random Forest, BiLSTM, RoBERTa)
- Pre-loaded example texts for quick testing
- Clean, intuitive design optimized for user experience

## 💡 Use Cases

- Academic integrity verification and plagiarism detection
- Content authenticity assessment for journalism and media
- Social media bot and synthetic content detection
- Quality control for human-authored content
- Research on LLM detection capabilities and limitations
- Educational tool for understanding AI writing patterns

## ⚠️ Limitations

- Detection accuracy decreases with very short texts (<50 words)
- May struggle with heavily edited or human-AI collaborative content
- Performance depends on similarity between training data and target AI models
- Future sophisticated LLMs may evade current detection methods
- Results should be interpreted as probabilistic evidence, not definitive proof

## 🛠️ Technologies

- **Python 3.10+** with PyTorch for deep learning
- **HuggingFace Transformers** for pre-trained language models (RoBERTa)
- **scikit-learn** for traditional ML algorithms (Logistic Regression, Random Forest)
- **NLTK & spaCy** for linguistic feature extraction and NLP preprocessing
- **LIME & SHAP** for model interpretability and explainable AI
- **Gradio** for interactive web interface
- **CUDA** for GPU-accelerated training (GTX 1660 Ti compatible)

*DetectAI is designed for research and educational purposes. Results are probabilistic and should be interpreted as supportive evidence rather than definitive proof. Detection accuracy may vary based on text characteristics and the specific AI models used for generation.*
