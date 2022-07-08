import torch
from torch import nn

from config import config
from utils import *

class TransformerDynamicsModel(nn.Module):
    def __init__(self, input_dim, global_dim):
        super().__init__()
        self.mlp_in = nn.Linear(input_dim, config.model.encoder.DIM)
        self.pos_enc = PositionalEncoding(config.model.encoder.DIM)
        self.encoder = TransformerEncoder(config.model.encoder.DIM)
        self.mlp_global = nn.Linear(global_dim, config.model.GLOBAL_EMB_DIM)
        self.decoder = Decoder(config.model.encoder.DIM + config.model.GLOBAL_EMB_DIM, input_dim)
    
    def forward(self, obs, obs_mask, global_obs):
        obs = self.mlp_in(obs)
        obs = self.pos_enc(obs)
        obs = self.encoder(obs, key_padding_mask=obs_mask)
        global_obs = self.mlp_global(global_obs)
        obs = torch.cat([])



class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(config.model.DROPOUT)
        self.position_encoding = nn.Parameter(torch.randn(config.model.MAX_SEQ_LEN, 1, input_dim))
        if batch_first:
            self.position_encoding = self.position_encoding.permute(1, 0, 2)
    
    def forward(self, x):
        x = x + self.position_encoding
        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = make_nn([input_dim, *config.model.decoder.HIDDEN_DIMS, output_dim])
    
    def forward(self, x):
        return self.mlp(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoders = nn.ModuleList([TransformerEncoderLayer(input_dim) for _ in config.model.encoder.N_LAYERS])
    
    def forward(self, x, key_padding_mask=None):
        for encoder in self.encoders:
            x = encoder(x, key_padding_mask=key_padding_mask)
        return x
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.in_norm = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(input_dim, config.model.encoder.N_HEADS, dropout=config.model.DROPOUT)
        self.attn_dropout = nn.Dropout(config.model.DROPOUT)
        self.attn_norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.model.encoder.FF_DIM),
            nn.ReLU(),
            nn.Dropout(config.model.DROPOUT),
            nn.Linear(config.model.encoder.FF_DIM, input_dim),
            nn.Dropout(config.model.DROPOUT)
        )
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x2 = self.in_norm(x)
        x2 = self.attn(x2, x2, x2, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.attn_dropout(x2)

        x2 = self.attn_norm(x)
        x2 = self.mlp(x2)

        return x + x2