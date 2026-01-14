import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=512):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx].lower().split()
        label = self.labels[idx]
        
        # Convert to indices
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in text]
        
        # Pad or truncate
        if len(indices) < self.max_len:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 2)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        output = self.dropout(attended)
        output = self.fc(output)
        
        return output

def build_vocab(texts, max_vocab=10000):
    """Build vocabulary from texts"""
    word_freq = {}
    for text in texts:
        for word in text.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in sorted_words[:max_vocab-2]:
        vocab[word] = len(vocab)
    
    return vocab

def train_bilstm():
    """Train BiLSTM model"""
    
    df = pd.read_csv('data/processed_data.csv')
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values, df['label'].values, 
        test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Build vocabulary
    vocab = build_vocab(X_train)
    joblib.dump(vocab, 'models/vocab.pkl')
    
    # Create datasets
    train_dataset = TextDataset(X_train, y_train, vocab)
    test_dataset = TextDataset(X_test, y_test, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = BiLSTMClassifier(vocab_size=len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    print("Training BiLSTM model...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\nBiLSTM Results:")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Human', 'AI']))
    
    # Save model
    torch.save(model.state_dict(), 'models/bilstm_model.pth')
    
    return model

if __name__ == "__main__":
    model = train_bilstm()
    print("\nBiLSTM training complete!")
