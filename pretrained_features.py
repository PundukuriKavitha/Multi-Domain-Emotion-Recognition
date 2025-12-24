"""
FIXED VERSION - Windows Compatible
Extracts features using pre-trained HuBERT/Wav2vec2 models
Uses librosa instead of torchaudio to avoid FFmpeg issues
"""

import os
import numpy as np
import torch
import librosa  # ← Using librosa instead of torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, HubertModel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIGURATION
# =========================
RAVDESS_PATH = "data/source/RAVDESS"
TESS_PATH = "data/target/TESS"
FEATURES_OUTPUT = "features_pretrained"
os.makedirs(FEATURES_OUTPUT, exist_ok=True)

EMOTIONS = ["angry", "happy", "sad", "neutral"]
SAMPLE_RATE = 16000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"🔧 Using device: {DEVICE}")

# =========================
# PRE-TRAINED MODEL OPTIONS
# =========================
PRETRAINED_MODELS = {
    'wav2vec2-base': {
        'model_name': 'facebook/wav2vec2-base-960h',
        'feature_dim': 768,
        'description': 'Wav2Vec2 Base (trained on 960h LibriSpeech)'
    },
    'wav2vec2-large': {
        'model_name': 'facebook/wav2vec2-large-960h',
        'feature_dim': 1024,
        'description': 'Wav2Vec2 Large (trained on 960h LibriSpeech)'
    },
    'hubert-base': {
        'model_name': 'facebook/hubert-base-ls960',
        'feature_dim': 768,
        'description': 'HuBERT Base (trained on 960h LibriSpeech)'
    },
    'hubert-large': {
        'model_name': 'facebook/hubert-large-ls960-ft',
        'feature_dim': 1024,
        'description': 'HuBERT Large (trained on 960h LibriSpeech, fine-tuned)'
    },
}

# =========================
# FEATURE EXTRACTOR CLASS
# =========================
class PretrainedFeatureExtractor:
    """Extract features using pre-trained speech models"""
    
    def __init__(self, model_type='wav2vec2-base', device='cuda'):
        self.device = device
        self.model_type = model_type
        self.model_config = PRETRAINED_MODELS[model_type]
        
        print(f"\n📥 Loading {self.model_config['description']}...")
        
        # Load feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_config['model_name']
        )
        
        # Load model
        if 'hubert' in model_type:
            self.model = HubertModel.from_pretrained(
                self.model_config['model_name']
            )
        else:
            self.model = Wav2Vec2Model.from_pretrained(
                self.model_config['model_name']
            )
        
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"✅ Model loaded: {self.model_config['feature_dim']}D features")
    
    def extract_features(self, audio_path, aggregation='mean'):
        """
        Extract features from audio file using librosa (FIXED for Windows)
        
        Parameters:
        -----------
        audio_path: str
            Path to audio file
        aggregation: str
            How to aggregate temporal features: 'mean', 'max', 'mean+std'
            
        Returns:
        --------
        features: numpy array
            Feature vector
        """
        try:
            # Load audio using librosa (THIS FIXES THE FFMPEG ERROR!)
            waveform, sample_rate = librosa.load(
                audio_path, 
                sr=SAMPLE_RATE,
                mono=True,
                dtype=np.float32
            )
            
            # Ensure waveform is the right shape
            if len(waveform.shape) > 1:
                waveform = waveform[0]  # Take first channel if stereo
            
            # Prepare input for model
            inputs = self.feature_extractor(
                waveform,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state  # Shape: (batch, time, features)
            
            # Aggregate temporal dimension
            if aggregation == 'mean':
                features = torch.mean(hidden_states, dim=1).squeeze()
            
            elif aggregation == 'max':
                features = torch.max(hidden_states, dim=1)[0].squeeze()
            
            elif aggregation == 'mean+std':
                mean_features = torch.mean(hidden_states, dim=1).squeeze()
                std_features = torch.std(hidden_states, dim=1).squeeze()
                features = torch.cat([mean_features, std_features], dim=0)
            
            else:
                features = torch.mean(hidden_states, dim=1).squeeze()
            
            return features.cpu().numpy()
        
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            feature_dim = self.model_config['feature_dim']
            if aggregation == 'mean+std':
                feature_dim *= 2
            return np.zeros(feature_dim)

# =========================
# PROCESS DATASET
# =========================
def process_dataset_pretrained(dataset_path, dataset_name, extractor, aggregation='mean'):
    """Extract pretrained features from all audio files"""
    data = []
    
    for emotion in EMOTIONS:
        emotion_path = os.path.join(dataset_path, emotion)
        if not os.path.exists(emotion_path):
            continue
        
        files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
        
        for file in tqdm(files, desc=f"{dataset_name} - {emotion}"):
            audio_path = os.path.join(emotion_path, file)
            features = extractor.extract_features(audio_path, aggregation)
            
            data.append({
                'features': features,
                'emotion': emotion,
                'dataset': dataset_name,
                'file': file
            })
    
    return data

# =========================
# MAIN EXECUTION
# =========================
def extract_all_pretrained_features(model_type='wav2vec2-base', aggregation='mean'):
    """
    Extract features using specified pre-trained model
    """
    print("="*70)
    print(f" 🎤 EXTRACTING FEATURES USING PRE-TRAINED MODELS")
    print("="*70)
    
    # Initialize extractor
    extractor = PretrainedFeatureExtractor(model_type=model_type, device=DEVICE)
    
    # Process RAVDESS (Source)
    print(f"\n📊 Processing RAVDESS (Source)...")
    ravdess_data = process_dataset_pretrained(
        RAVDESS_PATH, "RAVDESS", extractor, aggregation
    )
    
    # Process TESS (Target)
    print(f"\n📊 Processing TESS (Target)...")
    tess_data = process_dataset_pretrained(
        TESS_PATH, "TESS", extractor, aggregation
    )
    
    # Convert to arrays
    ravdess_X = np.array([d['features'] for d in ravdess_data])
    ravdess_y = np.array([d['emotion'] for d in ravdess_data])
    
    tess_X = np.array([d['features'] for d in tess_data])
    tess_y = np.array([d['emotion'] for d in tess_data])
    
    # Save features
    print(f"\n💾 Saving features...")
    
    output_suffix = f"{model_type}_{aggregation}"
    
    np.savez(
        os.path.join(FEATURES_OUTPUT, f'ravdess_{output_suffix}.npz'),
        X=ravdess_X, y=ravdess_y
    )
    np.savez(
        os.path.join(FEATURES_OUTPUT, f'tess_{output_suffix}.npz'),
        X=tess_X, y=tess_y
    )
    
    print(f"\n✅ Feature extraction complete!")
    print(f"   RAVDESS: {ravdess_X.shape[0]} samples × {ravdess_X.shape[1]} features")
    print(f"   TESS: {tess_X.shape[0]} samples × {tess_X.shape[1]} features")
    print(f"   Saved to: {FEATURES_OUTPUT}/")
    
    return ravdess_X, ravdess_y, tess_X, tess_y

# =========================
# EXTRACT WITH MULTIPLE MODELS
# =========================
def extract_all_model_features():
    """Extract features using all available models for comparison"""
    
    print("\n" + "="*70)
    print(" 🔬 EXTRACTING FEATURES FROM PRE-TRAINED MODELS")
    print("="*70)
    
    results = {}
    
    # Start with base models (faster)
    for model_type in ['wav2vec2-base', 'hubert-base']:
        print(f"\n{'='*70}")
        print(f" Processing: {PRETRAINED_MODELS[model_type]['description']}")
        print(f"{'='*70}")
        
        try:
            X_source, y_source, X_target, y_target = extract_all_pretrained_features(
                model_type=model_type,
                aggregation='mean'
            )
            
            results[model_type] = {
                'source': (X_source, y_source),
                'target': (X_target, y_target),
                'feature_dim': X_source.shape[1]
            }
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"❌ Error with {model_type}: {e}")
            continue
    
    print("\n" + "="*70)
    print(" ✅ ALL FEATURES EXTRACTED")
    print("="*70)
    
    for model_type, data in results.items():
        print(f"\n{model_type}:")
        print(f"  Feature dimension: {data['feature_dim']}")
        print(f"  Source samples: {data['source'][0].shape[0]}")
        print(f"  Target samples: {data['target'][0].shape[0]}")
    
    return results

# =========================
# RUN EXTRACTION
# =========================
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" 🎯 PRE-TRAINED MODEL FEATURE EXTRACTION (WINDOWS FIX)")
    print("="*70)
    
    print("\n📋 Available models:")
    for i, (key, value) in enumerate(PRETRAINED_MODELS.items(), 1):
        print(f"  {i}. {key}: {value['description']}")
        print(f"     Feature dimension: {value['feature_dim']}")
    
    print("\n" + "-"*70)
    print(" RECOMMENDED: wav2vec2-base (fastest, good accuracy)")
    print("-"*70)
    
    # Extract features
    print("\n🚀 Extracting features with wav2vec2-base...")
    extract_all_pretrained_features(model_type='wav2vec2-base', aggregation='mean')
    
    print("\n✅ Ready for training!")
    print("\nNext step: Run complete_pretrained_pipeline.py")