# audio_stream.py

import sounddevice as sd
import numpy as np

def audio_stream_generator(chunk_size=1024, sample_rate=16000):
    """Yield audio chunks from the microphone in real-time."""
    
    def callback(indata, frames, time, status):
        """This callback function will be called for every audio block captured."""
        if status:
            print(status)
        # Yield the audio data as chunks
        audio_chunk = np.copy(indata)  # Make a copy of the audio chunk
        yield audio_chunk

    # Open an audio stream with the specified chunk size and sample rate
    with sd.InputStream(callback=callback, blocksize=chunk_size, channels=1, samplerate=sample_rate):
        print(f"Listening to microphone at {sample_rate}Hz, chunk size: {chunk_size} samples")
        while True:
            # Continuously listen and yield the audio chunks
            pass  # Infinite loop to keep the stream open
