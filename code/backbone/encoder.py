import torch
import torch.nn as nn

class Transformer_Encoder(nn.Module):
    def __init__(self, input_size, hidden_dim, nlayers, dropout=0.5):
        super(Transformer_Encoder, self).__init__()
        
        self.input_size = input_size    
        
        nhead = 1
        for h in range(5, 15):
            if self.input_size % h == 0:
                nhead = h
                break
            
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dropout=dropout, batch_first=True)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        self.lin_out = torch.nn.Linear(input_size, hidden_dim, bias=True)
        
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.lin_out(x)
        return x