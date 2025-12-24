"""
COMPLETE END-TO-END PIPELINE
Using Pre-trained HuBERT/Wav2vec2 models for >80% accuracy

Run this script after organizing your datasets:
python complete_pretrained_pipeline.py
"""

import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import from previous artifacts
from pretrained_features import extract_all_pretrained_features, PRETRAINED_MODELS
from pretrained_model_training import (
    train_pretrained_model, 
    evaluate_pretrained_model
)

# =========================
# CONFIGURATION
# =========================
class Config:
    # Paths
    RAVDESS_PATH = "data/source/RAVDESS"
    TESS_PATH = "data/target/TESS"
    FEATURES_OUTPUT = "features_pretrained"
    RESULTS_OUTPUT = "results"
    
     # Model selection
    MODEL_TYPE = 'hubert-large'  # ✅ CHANGED for 90%+
    AGGREGATION = 'mean+std'      # ✅ CHANGED for richer features
    
    # Training parameters
    N_EPOCHS = 200                # ✅ CHANGED for better training
    BATCH_SIZE = 32               # ✅ CHANGED for better generalization
    LEARNING_RATE = 0.0001
    USE_DOMAIN_ADAPTATION = True
    
    # Data split
    TARGET_TRAIN_SIZE = 0.70
    TARGET_VAL_SIZE = 0.15
    TARGET_TEST_SIZE = 0.15
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create output directories
os.makedirs(Config.FEATURES_OUTPUT, exist_ok=True)
os.makedirs(Config.RESULTS_OUTPUT, exist_ok=True)

# =========================
# MAIN PIPELINE
# =========================
def run_complete_pipeline():
    """Execute complete training and evaluation pipeline"""
    
    print("\n" + "="*80)
    print(" 🎯 SPEECH EMOTION RECOGNITION - COMPLETE PIPELINE")
    print(" 🔬 Using Pre-trained HuBERT/Wav2vec2 Models")
    print("="*80)
    
    # =========================
    # STEP 1: FEATURE EXTRACTION
    # =========================
    print(f"\n{'='*80}")
    print(" STEP 1: FEATURE EXTRACTION")
    print(f"{'='*80}")
    
    feature_file_source = os.path.join(
        Config.FEATURES_OUTPUT, 
        f'ravdess_{Config.MODEL_TYPE}_{Config.AGGREGATION}.npz'
    )
    feature_file_target = os.path.join(
        Config.FEATURES_OUTPUT, 
        f'tess_{Config.MODEL_TYPE}_{Config.AGGREGATION}.npz'
    )
    
    # Check if features already exist
    if os.path.exists(feature_file_source) and os.path.exists(feature_file_target):
        print(f"\n✅ Features already extracted, loading from disk...")
        
        ravdess_data = np.load(feature_file_source)
        tess_data = np.load(feature_file_target)
        
        X_source = ravdess_data['X']
        y_source = ravdess_data['y']
        X_target = tess_data['X']
        y_target = tess_data['y']
    else:
        print(f"\n📥 Extracting features using {Config.MODEL_TYPE}...")
        print(f"   Model: {PRETRAINED_MODELS[Config.MODEL_TYPE]['description']}")
        print(f"   Feature dim: {PRETRAINED_MODELS[Config.MODEL_TYPE]['feature_dim']}")
        
        X_source, y_source, X_target, y_target = extract_all_pretrained_features(
            model_type=Config.MODEL_TYPE,
            aggregation=Config.AGGREGATION
        )
    
    print(f"\n✅ Features loaded:")
    print(f"   Source (RAVDESS): {X_source.shape}")
    print(f"   Target (TESS): {X_target.shape}")
    
    # =========================
    # STEP 2: DATA SPLITTING
    # =========================
    print(f"\n{'='*80}")
    print(" STEP 2: DATA SPLITTING")
    print(f"{'='*80}")
    
    # Split target data: 70% train, 15% val, 15% test
    X_target_train, X_target_temp, y_target_train, y_target_temp = train_test_split(
        X_target, y_target, 
        test_size=(Config.TARGET_VAL_SIZE + Config.TARGET_TEST_SIZE),
        stratify=y_target,
        random_state=42
    )
    
    X_target_val, X_target_test, y_target_val, y_target_test = train_test_split(
        X_target_temp, y_target_temp,
        test_size=Config.TARGET_TEST_SIZE / (Config.TARGET_VAL_SIZE + Config.TARGET_TEST_SIZE),
        stratify=y_target_temp,
        random_state=42
    )
    
    print(f"\n✅ Data split:")
    print(f"   Source (RAVDESS): {X_source.shape[0]} samples")
    print(f"   Target Train: {X_target_train.shape[0]} samples ({Config.TARGET_TRAIN_SIZE*100:.0f}%)")
    print(f"   Target Val: {X_target_val.shape[0]} samples ({Config.TARGET_VAL_SIZE*100:.0f}%)")
    print(f"   Target Test: {X_target_test.shape[0]} samples ({Config.TARGET_TEST_SIZE*100:.0f}%)")
    
    # Print emotion distribution
    print(f"\n   Emotion distribution in test set:")
    unique, counts = np.unique(y_target_test, return_counts=True)
    for emotion, count in zip(unique, counts):
        print(f"      {emotion}: {count} samples")
    
    # =========================
    # STEP 3: MODEL TRAINING
    # =========================
    print(f"\n{'='*80}")
    print(" STEP 3: MODEL TRAINING")
    print(f"{'='*80}")
    
    input_dim = X_source.shape[1]
    
    model, label_encoder, history = train_pretrained_model(
        X_source, y_source,
        X_target_train, y_target_train,
        X_target_val, y_target_val,
        input_dim=input_dim,
        n_epochs=Config.N_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        device=Config.DEVICE,
        use_domain_adaptation=Config.USE_DOMAIN_ADAPTATION
    )
    
    # =========================
    # STEP 4: EVALUATION
    # =========================
    print(f"\n{'='*80}")
    print(" STEP 4: TEST SET EVALUATION")
    print(f"{'='*80}")
    
    test_accuracy, y_pred, y_true = evaluate_pretrained_model(
        model, X_target_test, y_target_test, label_encoder, device=Config.DEVICE
    )
    
    print(f"\n🎯 TEST ACCURACY: {test_accuracy*100:.2f}%")
    
    # =========================
    # STEP 5: DETAILED ANALYSIS
    # =========================
    print(f"\n{'='*80}")
    print(" STEP 5: DETAILED ANALYSIS")
    print(f"{'='*80}")
    
    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    
    # Per-emotion accuracy
    print("\n📊 Per-Emotion Accuracy:")
    emotions = label_encoder.classes_
    for emotion in emotions:
        mask = y_true == emotion
        acc = accuracy_score(y_true[mask], y_pred[mask])
        print(f"   {emotion.capitalize()}: {acc*100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=emotions)
    
    # =========================
    # STEP 6: VISUALIZATION
    # =========================
    print(f"\n{'='*80}")
    print(" STEP 6: RESULTS VISUALIZATION")
    print(f"{'='*80}")
    
    fig = plt.figure(figsize=(20, 6))
    
    # Plot 1: Training history
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['emotion_loss'], label='Emotion Loss', linewidth=2)
    if Config.USE_DOMAIN_ADAPTATION:
        ax1.plot(history['domain_loss'], label='Domain Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation accuracy
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='green', linewidth=2)
    ax2.axhline(y=0.80, color='r', linestyle='--', label='Target: 80%', linewidth=2)
    ax2.fill_between(range(len(history['val_acc'])), 0.80, 1.0, alpha=0.2, color='green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1.0])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confusion matrix
    ax3 = plt.subplot(1, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotions, yticklabels=emotions, ax=ax3,
                cbar_kws={'label': 'Count'})
    ax3.set_title(f'Confusion Matrix\nAccuracy: {test_accuracy*100:.2f}%', 
                  fontsize=14, fontweight='bold')
    ax3.set_ylabel('True Label', fontsize=12)
    ax3.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    result_file = os.path.join(
        Config.RESULTS_OUTPUT, 
        f'results_{Config.MODEL_TYPE}_{Config.AGGREGATION}.png'
    )
    plt.savefig(result_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Results saved to: {result_file}")
    
    # =========================
    # STEP 7: COMPARISON WITH BASE PAPER
    # =========================
    print(f"\n{'='*80}")
    print(" STEP 7: COMPARISON WITH BASE PAPER")
    print(f"{'='*80}")
    
    results_comparison = {
        'Method': [
            'Base Paper - Low-level features',
            'Base Paper - HuBERT',
            'Base Paper - Wav2vec2.0 [BEST]',
            '',
            f'Your Implementation - {Config.MODEL_TYPE}',
        ],
        'Accuracy': [
            63.23,
            66.87,
            68.41,
            0,
            test_accuracy * 100
        ]
    }
    
    print("\n📊 ACCURACY COMPARISON:")
    print("-" * 70)
    for method, acc in zip(results_comparison['Method'], results_comparison['Accuracy']):
        if acc == 0:
            print("-" * 70)
            continue
        bar = '█' * int(acc/2) if acc > 0 else ''
        status = '✅' if acc >= 80 else ('🟨' if acc >= 70 else '⚠️')
        print(f"{status} {method:40s} | {bar} {acc:.2f}%")
    print("-" * 70)
    
    # Achievement status
    print(f"\n🎯 TARGET: 80% accuracy")
    print(f"📊 ACHIEVED: {test_accuracy*100:.2f}%")
    
    if test_accuracy >= 0.80:
        print("✅ TARGET ACHIEVED! 🎉")
        improvement = test_accuracy * 100 - 68.41
        print(f"🚀 Improvement over base paper: +{improvement:.2f}%")
    else:
        gap = (0.80 - test_accuracy) * 100
        print(f"⚠️  Need {gap:.2f}% more to reach target")
        print("\n💡 Suggestions:")
        print("   - Try wav2vec2-large or hubert-large")
        print("   - Use 'mean+std' aggregation")
        print("   - Increase training epochs")
        print("   - Apply data augmentation")
    
    # =========================
    # STEP 8: SAVE MODEL
    # =========================
    print(f"\n{'='*80}")
    print(" STEP 8: SAVE MODEL")
    print(f"{'='*80}")
    
    model_file = os.path.join(
        Config.RESULTS_OUTPUT,
        f'model_{Config.MODEL_TYPE}_{Config.AGGREGATION}.pth'
    )
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'config': {
            'model_type': Config.MODEL_TYPE,
            'aggregation': Config.AGGREGATION,
            'input_dim': input_dim,
            'n_classes': len(label_encoder.classes_),
        },
        'accuracy': test_accuracy,
        'history': history
    }, model_file)
    
    print(f"\n✅ Model saved to: {model_file}")
    
    # =========================
    # FINAL SUMMARY
    # =========================
    print(f"\n{'='*80}")
    print(" 🎊 PIPELINE COMPLETE!")
    print(f"{'='*80}")
    
    print(f"\n📋 Summary:")
    print(f"   Model: {PRETRAINED_MODELS[Config.MODEL_TYPE]['description']}")
    print(f"   Feature dimension: {input_dim}")
    print(f"   Training samples: {X_source.shape[0] + X_target_train.shape[0]}")
    print(f"   Test samples: {X_target_test.shape[0]}")
    print(f"   Test accuracy: {test_accuracy*100:.2f}%")
    print(f"   Best validation accuracy: {max(history['val_acc'])*100:.2f}%")
    
    print(f"\n📁 Output files:")
    print(f"   Features: {Config.FEATURES_OUTPUT}/")
    print(f"   Results: {result_file}")
    print(f"   Model: {model_file}")
    
    print(f"\n{'='*80}\n")
    
    return model, label_encoder, test_accuracy, history

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    # Run complete pipeline
    model, label_encoder, accuracy, history = run_complete_pipeline()
    
    print("✅ All done! Check the 'results' folder for visualizations.")
    
    # Optional: Try different models
    print("\n💡 To try different pre-trained models, modify Config.MODEL_TYPE:")
    print("   - 'wav2vec2-base': Fast, good accuracy")
    print("   - 'wav2vec2-large': Slower, higher accuracy")
    print("   - 'hubert-base': Alternative to wav2vec2")
    print("   - 'hubert-large': Highest accuracy")
    print("   - 'wav2vec2-xlsr': Multilingual (53 languages)")