# asr_inference.py
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from microphone_stream import microphone_stream_generator
from feature_extraction import extract_log_mel_spectrogram

def load_pretrained_model(model_name):
    """Load a pre-trained Wav2Vec2 model and processor."""
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()
    return model, processor

def stream_inference(model, processor, audio_stream, sample_rate=16000):
    """Run streaming inference on real-time audio."""
    all_predictions = []
    
    for i, chunk in enumerate(audio_stream):
        print(f"Processing chunk {i + 1}...")

        if isinstance(chunk, np.ndarray):
            chunk = torch.tensor(chunk, dtype=torch.float32)
        else:
            print(f"Error: Invalid audio chunk format at chunk {i + 1}")
            continue

        # Reshape chunk to match expected input dimensions
        chunk = chunk.squeeze(0)

        # Process audio chunk
        inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt", padding=True)

        # Forward pass through the model
        with torch.no_grad():
            logits = model(**inputs).logits

        # Decode the output tokens
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        print(f"Chunk {i + 1}: Transcription: {transcription}")

        all_predictions.append(transcription)

    return all_predictions

if __name__ == "__main__":
    # Load the pre-trained model and processor
    model_name = "facebook/wav2vec2-base-960h"
    model, processor = load_pretrained_model(model_name)

    # Start streaming audio from the microphone
    audio_stream = microphone_stream_generator(chunk_size=1024, sample_rate=16000)

    # Run real-time inference
    transcriptions = stream_inference(model, processor, audio_stream)

    print("Final transcription:", transcriptions)
