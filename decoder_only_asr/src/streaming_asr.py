# streaming_asr.py

import torch
from feature_extraction import extract_log_mel_spectrogram
from model import SimpleDecoderASR
from audio_stream import audio_stream_generator  # Import the audio streaming function

# Load the trained ASR model
def load_trained_model(model_path, input_dim, vocab_size):
    """Load the trained model from the saved state dictionary."""
    model = SimpleDecoderASR(input_dim=input_dim, vocab_size=vocab_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

# Inference function for streaming audio
def stream_inference(model, audio_stream, chunk_size=1024):
    """Run streaming inference on audio chunks."""
    memory = None  # Initialize memory (for LSTM hidden states or transformer memory)
    partial_output = []
    
    for chunk in audio_stream:
        # Process the audio chunk: Extract log-Mel spectrogram
        features = extract_log_mel_spectrogram(chunk)
        features = torch.tensor(features).unsqueeze(0)  # Add batch dimension
        
        # Forward pass through the model
        with torch.no_grad():  # No need to calculate gradients for inference
            # If using LSTM, pass the memory (hidden state)
            output, memory = model(features, memory)
        
        # Decode the output tokens (using argmax for simplicity)
        predicted_tokens = torch.argmax(output, dim=-1)
        partial_output.append(predicted_tokens)
    
    # Concatenate the partial outputs
    final_output = torch.cat(partial_output, dim=0)
    return final_output

if __name__ == "__main__":
    # Define the path to the trained model
    model_path = 'asr_model_final.pth'

    # Define the input and vocab dimensions (must match the training setup)
    input_dim = 128  # Number of mel-spectrogram features
    vocab_size = 256  # Vocabulary size (same as training)

    # Load the trained model
    model = load_trained_model(model_path, input_dim, vocab_size)

    # Capture and process streaming audio
    chunk_size = 1024
    sample_rate = 16000
    audio_stream = audio_stream_generator(chunk_size, sample_rate)

    for audio_chunk in audio_stream:
        predicted_output = stream_inference(model, [audio_chunk], chunk_size)
        print(f"Predicted transcription: {predicted_output}")
