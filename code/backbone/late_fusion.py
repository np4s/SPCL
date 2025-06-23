import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import batch_flating
from .encoder import Transformer_Encoder

class LateFusion(nn.Module):
    def __init__(self, args):
        super(LateFusion, self).__init__()
        
        self.label_dict = args.dataset_label_dict[args.dataset]
        
        tag_size = len(self.label_dict)
        self.num_modal = len(args.modalities)
        self.args = args
        self.modalities = args.modalities
        self.embedding_dim = args.embedding_dim[args.dataset]
        self.device = args.device
        self.hidden_dim = args.hidden_dim
        self.dropout = args.drop_rate
        
        self.encoder = nn.ModuleDict()
        self.uni_fc = nn.ModuleDict()
        
        for m in self.modalities:
            self.encoder.add_module(m, self.get_encoder(m))
            self.uni_fc.add_module(m, nn.Linear(self.hidden_dim, tag_size))
            
    def get_encoder(self, m):
        print(f"Net --> {m} encoder: {self.args.encoder_modules}")
        if self.args.encoder_modules == "transformer":
            return Transformer_Encoder(
                self.embedding_dim[m], 
                self.hidden_dim,
                self.args.encoder_nlayers,
                dropout=self.dropout)
        if self.args.encoder_modules == "mamba":
            return Mamba_Encoder(
                self.embedding_dim[m], 
                self.hidden_dim, 
                d_state=self.args.d_state)
        
    def forward(self, data):
        x = data["tensor"]
        lengths = data["length"]

        encoded = {}
        logit = {}
        
        for m in self.modalities:
            encoded[m] = self.encoder[m](x[m])
            logit[m] = self.uni_fc[m](encoded[m])
            
        logit = batch_flating(logit, lengths, self.modalities, self.device)
        joint = torch.stack([logit[m] for m in self.modalities], dim=0).sum(dim=0)

        return joint, logit
    
    def get_encoder_params(self):
        params_dict = {
            m:[] for m in self.modalities
        }
    
        for m in self.modalities:
            for params in self.encoder[m].parameters():
                params_dict[m].append(params)
        return params_dict