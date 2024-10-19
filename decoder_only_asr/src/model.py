# # src/model.py
import torch
import torch.nn as nn

class SimpleDecoderASR(nn.Module):
    """Simple Transformer decoder model for ASR."""
    def __init__(self, input_dim, vocab_size, num_heads=4, hidden_dim=256, num_layers=2):
        super(SimpleDecoderASR, self).__init__()
        
        # Linear layer to project input features to hidden dimension
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer decoder
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        
        # Output layer that converts transformer outputs to token logits
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        """Forward pass through the decoder."""
        # x has shape (batch_size, n_mels, num_frames)
        
        # Transpose the input to (batch_size, num_frames, n_mels) before feeding into Linear layer
        x = x.transpose(1, 2)  # Shape becomes (batch_size, num_frames, n_mels)
        
        # Project the input features to hidden dimension
        x = self.embedding(x)  # Shape becomes (batch_size, num_frames, hidden_dim)
        
        # Apply the transformer decoder with self-attention (no memory required)
        output = self.transformer_decoder(x, x)  # Self-attention mechanism
        
        # Output token probabilities
        return self.fc_out(output)

# import torch
# import torch.nn as nn

# class SimpleDecoderASR(nn.Module):
#     """Simple Transformer decoder model for ASR."""
#     def __init__(self, input_dim, vocab_size, num_heads=4, hidden_dim=256, num_layers=2):
#         super(SimpleDecoderASR, self).__init__()
        
#         # Linear layer to project input features to hidden dimension
#         self.embedding = nn.Linear(input_dim, hidden_dim)
        
#         # Transformer decoder
#         self.transformer_decoder = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads),
#             num_layers=num_layers
#         )
        
#         # Output layer that converts transformer outputs to token logits
#         self.fc_out = nn.Linear(hidden_dim, vocab_size)
    
#     def forward(self, x, memory):
#         """Forward pass through the decoder."""
#         # x has shape (batch_size, n_mels, num_frames)
        
#         # Transpose the input to (batch_size, num_frames, n_mels) before feeding into Linear layer
#         x = x.transpose(1, 2)  # Shape becomes (batch_size, num_frames, n_mels)
        
#         # Project the input features to hidden dimension
#         x = self.embedding(x)  # Shape becomes (batch_size, num_frames, hidden_dim)
        
#         # Apply the transformer decoder
#         output = self.transformer_decoder(x, memory)
        
#         # Output token probabilities
#         return self.fc_out(output)
