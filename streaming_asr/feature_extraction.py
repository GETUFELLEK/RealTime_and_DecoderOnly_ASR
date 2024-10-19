# feature_extraction.py
import torchaudio
import torch

def extract_log_mel_spectrogram(waveform, sample_rate=16000):
    """Extract log-mel spectrogram features from waveform."""
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, 
        n_fft=400, 
        win_length=400, 
        hop_length=160, 
        n_mels=128
    )
    mel_spectrogram = transform(waveform)
    log_mel_spectrogram = torch.log(mel_spectrogram + 1e-6)
    return log_mel_spectrogram
