Here is a `README.md` file for the project:

---

# Real-Time and Decoder-Only ASR Projects

This repository contains two Automatic Speech Recognition (ASR) projects: a real-time streaming ASR and a decoder-only ASR system, both implemented in Python. The real-time ASR system utilizes a pre-trained Wav2Vec2 model from Hugging Face for speech-to-text conversion, while the decoder-only ASR system is built from scratch with custom training and inference capabilities.

## Project Structure

```
RealTime_and_DecoderOnly_ASR/
│
├── decoder_only_asr/
│   ├── data/                      # Contains dataset used for training
│   ├── asr_model_final.pth         # Saved model checkpoint (ignored in Git)
│   ├── src/
│   │   ├── __init__.py             # Initialization file for the package
│   │   ├── feature_extraction.py   # Extracts log-Mel spectrogram features
│   │   ├── model.py                # Defines the SimpleDecoderASR model architecture
│   │   ├── train.py                # Training script for the decoder-only ASR
│   │   ├── inference.py            # Inference script for ASR model
│   │   ├── streaming_asr.py        # Handles the streaming-based ASR tasks
│   │   ├── audio_stream.py         # Handles audio stream processing
│   └── asr_venv/                   # Python virtual environment (ignored in Git)
│
├── streaming_asr/
│   ├── asr_inference.py            # Inference script using pre-trained model
│   ├── feature_extraction.py       # Extracts log-Mel spectrogram features
│   ├── microphone_stream.py        # Captures audio from the microphone
│   ├── pretrained_model/           # Pre-trained Wav2Vec2 model from Hugging Face (ignored in Git)
│   ├── requirements.txt            # Dependencies required for the project
│   └── asr_venv/                   # Python virtual environment (ignored in Git)
│
├── .gitignore                      # Files and directories to be ignored in Git
└── README.md                       # Project documentation (this file)
```

## Project Overview

### 1. Decoder-Only ASR Project

The decoder-only ASR system is built from scratch using PyTorch and Torchaudio. It utilizes a custom model (SimpleDecoderASR) trained on the LibriSpeech dataset. The model is trained with a CTC loss function, and the extracted log-Mel spectrogram features serve as input to the model. 

#### Key Files:
- **src/train.py**: Handles model training, dataset loading, and saving checkpoints.
- **src/inference.py**: Performs inference using the trained decoder-only model on input audio.
- **src/feature_extraction.py**: Extracts log-Mel spectrogram features from audio files for both training and inference.
- **src/model.py**: Defines the model architecture for the ASR decoder.

### 2. Real-Time Streaming ASR Project

The real-time streaming ASR system uses a pre-trained Wav2Vec2 model from Hugging Face's transformers library. The system captures audio from a microphone in real-time, processes the audio in chunks, and converts it to text using the pre-trained model.

#### Key Files:
- **asr_inference.py**: This file contains the main inference logic that loads the pre-trained Wav2Vec2 model, processes audio streams, and transcribes speech.
- **microphone_stream.py**: Captures audio from a microphone and processes it in real-time for streaming inference.
- **feature_extraction.py**: Extracts the necessary features (log-Mel spectrogram) from the audio input.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/GETUFELLEK/RealTime_and_DecoderOnly_ASR.git
   cd RealTime_and_DecoderOnly_ASR
   ```

2. **Create virtual environments (optional but recommended):**

   For each project (`decoder_only_asr` and `streaming_asr`), create and activate a virtual environment.

   ```bash
   cd decoder_only_asr
   python3 -m venv asr_venv
   source asr_venv/bin/activate
   ```

3. **Install dependencies:**

   For each project, install the necessary dependencies from `requirements.txt`:

   ```bash
   pip install -r streaming_asr/requirements.txt
   ```

4. **Install additional libraries:**

   You may also need to install other dependencies like `sounddevice` and `transformers` manually if not already covered in `requirements.txt`:

   ```bash
   pip install sounddevice transformers torchaudio
   ```

## How to Run

### 1. Running the Decoder-Only ASR System

You can train the decoder-only ASR model using:

```bash
python src/train.py
```

To run inference on the trained model:

```bash
python src/inference.py
```

### 2. Running the Real-Time Streaming ASR System

For real-time inference using the pre-trained Wav2Vec2 model:

```bash
python streaming_asr/asr_inference.py
```

This will start capturing audio from the microphone, process it in real-time, and transcribe it to text.

## Model Training and Performance

- **Training**: The decoder-only ASR model is trained on the LibriSpeech dataset using the CTC loss function.
- **Inference**: Both models (decoder-only and pre-trained) handle real-time audio input for ASR tasks. The real-time model uses a pre-trained Wav2Vec2 model for more accurate and faster transcription.
  
## Future Improvements

- Improve the decoder-only ASR model’s accuracy by fine-tuning the model, training on larger datasets, and optimizing hyperparameters.
- Add more advanced feature extraction techniques for better audio representation.
- Expand the real-time ASR capabilities by integrating noise reduction or multi-lingual models.

---

This `README.md` provides an overview of the ASR projects, how to install and run them, and the future improvements that could be made to enhance both systems.
