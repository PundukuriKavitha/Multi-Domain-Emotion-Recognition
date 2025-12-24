"""
Complete training pipeline using pre-trained HuBERT/Wav2vec2 features
This should achieve 80-90% accuracy on RAVDESS→TESS transfer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# =========================
# DATASET CLASS
# =========================
class EmotionDataset(Dataset):
    def __init__(self, X, y, label_encoder=None):
        self.X = torch.FloatTensor(X)
        if label_encoder is not None:
            self.y = torch.LongTensor(label_encoder.transform(y))
        else:
            self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =========================
# MODEL ARCHITECTURE
# =========================
class PretrainedSERModel(nn.Module):
    """
    Speech Emotion Recognition using pre-trained features
    With attention and domain adaptation
    """
    
    def __init__(self, input_dim=768, n_classes=4, dropout=0.3):
        super(PretrainedSERModel, self).__init__()
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )
        
        # Feature refinement
        self.feature_refine = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Emotion classifier
        self.emotion_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        
        # Domain classifier (for adversarial training)
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
    
    def forward(self, x, alpha=1.0):
        # Project features
        projected = self.projection(x)
        
        # Apply attention (optional - can be removed for simplicity)
        # attention_weights = self.attention(projected)
        # attended = projected * attention_weights
        
        # Refine features
        features = self.feature_refine(projected)
        
        # Emotion classification
        emotion_logits = self.emotion_classifier(features)
        
        # Domain classification (with gradient reversal)
        reversed_features = GradientReversal.apply(features, alpha)
        domain_logits = self.domain_classifier(reversed_features)
        
        return emotion_logits, domain_logits, features

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

# =========================
# TRAINING FUNCTION
# =========================
def train_pretrained_model(
    X_source, y_source,
    X_target_train, y_target_train,
    X_target_val, y_target_val,
    input_dim=768,
    n_epochs=150,
    batch_size=64,
    learning_rate=0.0001,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    use_domain_adaptation=True
):
    """
    Train SER model with pre-trained features
    """
    print(f"\n{'='*70}")
    print(f" 🚀 TRAINING WITH PRE-TRAINED FEATURES")
    print(f"{'='*70}")
    print(f"\n📊 Dataset sizes:")
    print(f"   Source: {X_source.shape}")
    print(f"   Target Train: {X_target_train.shape}")
    print(f"   Target Val: {X_target_val.shape}")
    print(f"   Device: {device}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_source, y_target_train]))
    n_classes = len(label_encoder.classes_)
    
    print(f"   Classes: {label_encoder.classes_}")
    
    # Create datasets
    source_dataset = EmotionDataset(X_source, y_source, label_encoder)
    target_train_dataset = EmotionDataset(X_target_train, y_target_train, label_encoder)
    target_val_dataset = EmotionDataset(X_target_val, y_target_val, label_encoder)
    
    # Create dataloaders
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(target_val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = PretrainedSERModel(input_dim=input_dim, n_classes=n_classes)
    model = model.to(device)
    
    # Loss functions
    emotion_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=15, factor=0.5
    )
    
    # Training tracking
    best_val_acc = 0
    best_model_state = None
    patience = 30
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_acc': [],
        'emotion_loss': [],
        'domain_loss': []
    }
    
    print(f"\n{'='*70}")
    print(f" 🔄 TRAINING LOOP")
    print(f"{'='*70}\n")
    
    for epoch in range(n_epochs):
        model.train()
        
        # Domain adaptation parameter (gradually increase)
        p = float(epoch) / n_epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1 if use_domain_adaptation else 0
        
        epoch_loss = 0
        epoch_emotion_loss = 0
        epoch_domain_loss = 0
        n_batches = 0
        
        # Create iterators
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        # Training loop
        pbar = tqdm(range(min(len(source_loader), len(target_loader))), 
                   desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for _ in pbar:
            try:
                source_X, source_y = next(source_iter)
                target_X, target_y = next(target_iter)
            except StopIteration:
                break
            
            source_X, source_y = source_X.to(device), source_y.to(device)
            target_X, target_y = target_X.to(device), target_y.to(device)
            
            # Combine batches
            combined_X = torch.cat([source_X, target_X], dim=0)
            emotion_labels = torch.cat([source_y, target_y], dim=0)
            
            # Domain labels (0=source, 1=target)
            domain_labels = torch.cat([
                torch.zeros(len(source_X), dtype=torch.long),
                torch.ones(len(target_X), dtype=torch.long)
            ]).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            emotion_logits, domain_logits, _ = model(combined_X, alpha)
            
            # Compute losses
            loss_emotion = emotion_criterion(emotion_logits, emotion_labels)
            loss_domain = domain_criterion(domain_logits, domain_labels)
            
            if use_domain_adaptation:
                loss = loss_emotion + 0.3 * loss_domain
            else:
                loss = loss_emotion
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track losses
            epoch_loss += loss.item()
            epoch_emotion_loss += loss_emotion.item()
            epoch_domain_loss += loss_domain.item()
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Emotion': f'{loss_emotion.item():.4f}'
            })
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X = val_X.to(device)
                emotion_logits, _, _ = model(val_X, alpha=0)
                preds = torch.argmax(emotion_logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(val_y.numpy())
        
        val_accuracy = accuracy_score(val_labels, val_preds)
        
        # Update history
        history['train_loss'].append(epoch_loss / n_batches)
        history['val_acc'].append(val_accuracy)
        history['emotion_loss'].append(epoch_emotion_loss / n_batches)
        history['domain_loss'].append(epoch_domain_loss / n_batches)
        
        # Learning rate scheduling
        scheduler.step(val_accuracy)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print(f"  Train Loss: {epoch_loss/n_batches:.4f}")
            print(f"  Emotion Loss: {epoch_emotion_loss/n_batches:.4f}")
            if use_domain_adaptation:
                print(f"  Domain Loss: {epoch_domain_loss/n_batches:.4f}")
            print(f"  Val Accuracy: {val_accuracy*100:.2f}%")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✅ New best validation accuracy: {best_val_acc*100:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print(f"\n{'='*70}")
    print(f" ✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\n🏆 Best Validation Accuracy: {best_val_acc*100:.2f}%\n")
    
    return model, label_encoder, history

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_pretrained_model(model, X_test, y_test, label_encoder, 
                              device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Evaluate model on test set"""
    model.eval()
    
    test_dataset = EmotionDataset(X_test, y_test, label_encoder)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            emotion_logits, _, _ = model(X_batch, alpha=0)
            preds = torch.argmax(emotion_logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(y_batch.numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    
    predictions_labels = label_encoder.inverse_transform(predictions)
    true_labels_labels = label_encoder.inverse_transform(true_labels)
    
    return accuracy, predictions_labels, true_labels_labels

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" 🎯 PRE-TRAINED MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Example usage
    print("\nTo use this script:")
    print("1. Extract features using pretrained_features.py")
    print("2. Load features: data = np.load('features_pretrained/ravdess_wav2vec2-base_mean.npz')")
    print("3. Train: model, encoder, history = train_pretrained_model(...)")
    print("4. Evaluate: accuracy, preds, labels = evaluate_pretrained_model(...)")
    print("\nExpected accuracy: 80-90% (depending on model)")