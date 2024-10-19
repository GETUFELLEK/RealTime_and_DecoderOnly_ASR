import torch
import torchaudio
import numpy as np
import os
from feature_extraction import extract_log_mel_spectrogram
from model import SimpleDecoderASR

def load_trained_model(model_path, input_dim, vocab_size):
    """Load the trained model from the saved state dictionary."""
    print(f"Loading the trained model from {model_path}...")
    
    # Initialize the model architecture
    model = SimpleDecoderASR(input_dim=input_dim, vocab_size=vocab_size)
    
    # Load the trained model weights
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Set the model to evaluation mode (important for inference)
    model.eval()
    return model

def stream_inference(model, audio_stream, chunk_size=1024):
    """Run streaming inference on audio chunks."""
    print("Starting streaming inference...")
    partial_output = []
    
    for i, chunk in enumerate(audio_stream):
        print(f"Processing chunk {i + 1}...")

        # Convert the chunk (which is a NumPy array) to a PyTorch tensor if needed
        if isinstance(chunk, np.ndarray):
            chunk_tensor = torch.tensor(chunk).float()  # Convert chunk to torch tensor
        else:
            print(f"Error: Invalid audio chunk format at chunk {i + 1}")
            continue
        
        # Process the audio chunk: Extract log-Mel spectrogram
        try:
            features = extract_log_mel_spectrogram(chunk_tensor)  # Pass tensor

            # Convert features to PyTorch tensor if it is not already
            if isinstance(features, np.ndarray):
                features = torch.tensor(features).float()
                
            features = features.unsqueeze(0)  # Add batch dimension
            print(f"Chunk {i + 1}: Features extracted.")
        except Exception as e:
            print(f"Error extracting features from chunk {i + 1}: {e}")
            continue
        
        # Forward pass through the model
        with torch.no_grad():  # No need to calculate gradients for inference
            try:
                output = model(features)  # Only pass `features`, remove `memory`
                print(f"Chunk {i + 1}: Forward pass completed.")
            except Exception as e:
                print(f"Error during model forward pass at chunk {i + 1}: {e}")
                continue
        
        # Decode the output tokens (using argmax for simplicity)
        predicted_tokens = torch.argmax(output, dim=-1)
        print(f"Chunk {i + 1}: Predicted tokens: {predicted_tokens}")
        partial_output.append(predicted_tokens)
    
    # Concatenate the partial outputs into a final result
    try:
        final_output = torch.cat(partial_output, dim=0)
        print("Inference completed successfully.")
        print(f"Final output: {final_output}")
    except Exception as e:
        print(f"Error concatenating output: {e}")
        return None
    
    return final_output



def load_audio_as_stream(file_path, chunk_size=1024):
    """Simulate an audio stream by loading an audio file and splitting it into chunks."""
    waveform, sample_rate = torchaudio.load(file_path)  # Load audio file
    
    # Simulate streaming by splitting the waveform into chunks
    num_chunks = waveform.size(1) // chunk_size
    for i in range(num_chunks):
        yield waveform[:, i * chunk_size: (i + 1) * chunk_size].numpy()

if __name__ == "__main__":
    # Define the path to the trained model
    model_path = 'asr_model_final.pth'

    # Define the input and vocab dimensions (must match the training setup)
    input_dim = 128  # Number of mel-spectrogram features
    vocab_size = 256  # Vocabulary size (same as training)

    # Load the trained model
    print("Loading model...")
    model = load_trained_model(model_path, input_dim, vocab_size)
    
    if model is None:
        print("Failed to load model. Exiting...")
        exit(1)

    # Define the path to the audio file
    audio_file_path = '/mnt/c/Users/getne/Projects_2024/AI_projects_2024/decoder_only_asr/short_speech.wav'

    # Check if the audio file exists
    if os.path.exists(audio_file_path):
        print(f"Audio file found: {audio_file_path}")
        audio_stream = load_audio_as_stream(audio_file_path, chunk_size=1024)
    else:
        print(f"Audio file not found: {audio_file_path}. Exiting...")
        exit(1)

    # Run inference on the streaming audio
    print("Starting inference...")
    predicted_output = stream_inference(model, audio_stream)
    
    if predicted_output is not None:
        print(f"Predicted transcription: {predicted_output}")
    else:
        print("Inference failed.")
