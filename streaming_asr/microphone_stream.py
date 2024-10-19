# microphone_stream.py
import sounddevice as sd
import numpy as np

def microphone_stream_generator(chunk_size=1024, sample_rate=16000):
    """Stream live audio from the microphone."""
    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_queue.append(indata.copy())

    audio_queue = []
    
    with sd.InputStream(callback=callback, blocksize=chunk_size, channels=1, samplerate=sample_rate):
        while True:
            if audio_queue:
                yield audio_queue.pop(0)
