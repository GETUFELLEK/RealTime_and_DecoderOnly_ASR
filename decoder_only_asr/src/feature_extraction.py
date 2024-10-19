# # src/feature_extraction.py

import torchaudio
import librosa
import numpy as np

# src/feature_extraction.py

import torchaudio
import librosa
import numpy as np

def extract_log_mel_spectrogram(waveform, sample_rate=16000, n_mels=128):
    """Extract log-mel spectrogram directly from the waveform tensor."""
    # Convert waveform (tensor) to NumPy array for librosa processing
    waveform = waveform.numpy()[0]  
    
    # Ensure consistent number of mel bins (n_mels)
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels)
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec

