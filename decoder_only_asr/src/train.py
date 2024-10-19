# src/train.py

import torch
import torch.optim as optim
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from model import SimpleDecoderASR  # Import your model definition
from feature_extraction import extract_log_mel_spectrogram  # Assume this is your custom feature extraction

# Dataset Loader: Loading LibriSpeech dataset
class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, url="test-clean"):
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=root_dir, url=url, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, utterance, _, _, _ = self.dataset[idx]
        
        # Process the waveform using log-mel spectrogram
        log_mel_spectrogram = extract_log_mel_spectrogram(waveform, sample_rate)
        
        # Convert utterance (text) to token IDs for training (simplified tokenization)
        tokens = [ord(c) for c in utterance]  # Consider replacing with a more advanced tokenizer if needed
        return torch.tensor(log_mel_spectrogram, dtype=torch.float32), torch.tensor(tokens)


# Custom collate function to handle variable-length inputs and targets
def collate_fn(batch):
    inputs, targets = zip(*batch)

    # Ensure inputs are all tensors and have consistent frequency dimensions
    inputs = [torch.tensor(x, dtype=torch.float32) for x in inputs]

    # Get the number of mel bins from the first sample (should be consistent across all inputs)
    num_mel_bins = inputs[0].shape[0]

    # Find the maximum number of frames (time steps) in the batch
    max_frames = max([x.shape[1] for x in inputs])

    # Pad inputs along the time axis (dimension 1)
    padded_inputs = []
    input_lens = []
    for input_tensor in inputs:
        input_lens.append(input_tensor.shape[1])  # Original number of frames for each sample
        if input_tensor.shape[1] < max_frames:
            pad_amount = max_frames - input_tensor.shape[1]
            padding = torch.zeros((num_mel_bins, pad_amount), dtype=torch.float32)
            padded_input = torch.cat((input_tensor, padding), dim=1)
        else:
            padded_input = input_tensor
        padded_inputs.append(padded_input)

    # Stack the padded spectrograms into a batch tensor
    inputs = torch.stack(padded_inputs)

    # Pad targets (utterances/tokens) and get their lengths
    target_lens = [len(t) for t in targets]
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

    # Return inputs, targets, and their lengths
    return inputs, targets, torch.tensor(input_lens, dtype=torch.long), torch.tensor(target_lens, dtype=torch.long)


# Main training function
def train_asr_model(root_dir, vocab_size, input_dim, num_epochs=10, batch_size=8):
    # Initialize the dataset and dataloader
    train_dataset = ASRDataset(root_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize model, loss function (CTC), and optimizer
    model = SimpleDecoderASR(input_dim=input_dim, vocab_size=vocab_size)
    criterion = nn.CTCLoss(blank=vocab_size - 1)  # CTC loss with the last token as blank
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets, input_lens, target_lens in train_loader:
            optimizer.zero_grad()

            # Forward pass through the model
            outputs = model(inputs)  # Shape: (batch_size, seq_length, vocab_size)

            # Compute loss (CTC loss)
            loss = criterion(outputs.transpose(0, 1), targets, input_lens, target_lens)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model weights
            
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss}")

        # Save the model after each epoch (optional)
        torch.save(model.state_dict(), f'asr_model_epoch_{epoch + 1}.pth')
        print(f"Model saved after epoch {epoch + 1}")

    # Save the final model
    torch.save(model.state_dict(), 'asr_model_final.pth')
    print('Final model saved successfully!')


if __name__ == "__main__":
    # Directory where dataset will be downloaded
    root_dir = "./data"

    # Define model input/output dimensions
    input_dim = 128  # Number of mel-spectrogram features
    vocab_size = 256  # Example: ASCII character set size (this can change based on your tokenization)

    # Start training the model
    train_asr_model(root_dir=root_dir, vocab_size=vocab_size, input_dim=input_dim, num_epochs=10, batch_size=8)
