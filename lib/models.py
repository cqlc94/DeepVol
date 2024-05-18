import math
import torch
from torch import nn

eps = 1e-8
class RNN(nn.Module):
    def __init__(self, d_in=1, d_hidden=64, n_layers=1, dropout=0.2, **kwargs):
        super().__init__()
        self.rnn = nn.RNN(d_in, d_hidden, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_hidden, 1)
        self.softplus = nn.Softplus()

    def forward(self, x): # (B, d_in, T)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        return self.softplus(self.linear(x)) + eps
    
class GRU(nn.Module):
    def __init__(self, d_in=1, d_hidden=64, n_layers=2, dropout=0.2, **kwargs):
        super().__init__()
        self.gru = nn.GRU(d_in, d_hidden, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_hidden, 1)
        self.softplus = nn.Softplus()

    def forward(self, x): # (B, d_in, T)
        x, _ = self.gru(x)
        x = self.dropout(x)
        return self.softplus(self.linear(x)) + eps

class LSTM(nn.Module):
    def __init__(self, d_in=1, d_hidden=64, n_layers=1, dropout=0.2, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(d_in, d_hidden, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_hidden, 1)
        self.softplus = nn.Softplus()

    def forward(self, x): # (B, d_in, T)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return self.softplus(self.linear(x)) + eps
    

# class Transformer
class Transformer(nn.Module):
    def __init__(self, d_in=4, d_hidden=64, nhead=2, d_ff=256, n_layers=2, dropout=0.1, **kwargs):
        super().__init__()
        self.embedding = nn.Linear(d_in, d_hidden)
        self.pos_enc = PositionalEncoding(d_hidden, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_hidden, nhead=nhead, batch_first=True,
                                                   dim_feedforward=d_ff, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.linear_out = nn.Linear(d_hidden, 1) 
        self.softplus = nn.Softplus()       

    def forward(self, x, padding_mask):
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x, 
                                     mask=generate_self_atten_mask(x.size(1), x.device),
                                     src_key_padding_mask=padding_mask)
        return self.softplus(self.linear_out(x).squeeze(-1)) + eps

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=8000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
def generate_self_atten_mask(sz, device):
    return torch.tril(torch.ones(sz, sz, device=device))==0

class GARCH(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Parameters initialized in log space to ensure positivity
        self.omega_log = nn.Parameter(torch.tensor(0.0))
        self.alpha_log = nn.Parameter(torch.tensor(0.0))
        self.beta_log = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        T = x.size(1)
        sigma2 = [x.var(dim=1)]
        
        # Convert parameters back to their original space
        omega = torch.exp(self.omega_log)
        alpha = torch.exp(self.alpha_log) / (1.0 + torch.exp(self.alpha_log) + torch.exp(self.beta_log))
        beta = torch.exp(self.beta_log) / (1.0 + torch.exp(self.alpha_log) + torch.exp(self.beta_log))
        
        for t in range(1, T):
            sigma2.append(omega + alpha * x[:, t-1]**2 + beta * sigma2[t-1])

        sigma2 = torch.stack(sigma2, dim=1).squeeze(-1)**0.5
        return sigma2

    








