from torch.nn import RNN,Linear,LayerNorm,LeakyReLU
from torch import nn,Tensor
import torch
import math

class BasicRNN(torch.nn.Module):
    def __init__(self, embedding_dim:int,
                 vocab_size:int,
                 hidden_size:int,
                 num_layers:int,
                 n_meta:int):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        #self.num_layers_meta=num_layers_meta
        self.n_meta=n_meta

        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.sequence_rnn=RNN(embedding_dim,hidden_size,num_layers,batch_first=True)
        self.meta_network=torch.nn.Sequential(
            *[Linear(hidden_size,hidden_size//2),
              LayerNorm(hidden_size//2),
              LeakyReLU(),
              Linear(hidden_size//2, n_meta*2),
              LayerNorm(n_meta*2),
              LeakyReLU(),
              Linear(n_meta*2,n_meta)
              ]
        )


    def forward(self,input_batch):
        #input_batch (B,L)
        embedded_batch=self.embedding(input_batch)
        rnn_output,rnn_h=self.sequence_rnn(embedded_batch) #(B,L,H_in) #(num_layers, B,hidden_size)

        hidden=rnn_h[0].squeeze(0) #(B, hidden_size)
        meta=self.meta_network(hidden)

        return meta
        

class PositionalEncoding(torch.nn.Module):

    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class BasicTransformer(torch.nn.Module):
    def __init__(self, embedding_dim,vocab_size,nhead,num_layers,n_meta):
        super().__init__()
        self.positional_encoding=PositionalEncoding(embedding_dim)
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,nhead=nhead)
        self.encoder=nn.TransformerEncoder(encoder_layer,num_layers)

        self.meta_network=torch.nn.Sequential(
            *[Linear(embedding_dim,embedding_dim//2),
              LayerNorm(embedding_dim//2),
              LeakyReLU(),
              Linear(embedding_dim//2, n_meta*2),
              LayerNorm(n_meta*2),
              LeakyReLU(),
              Linear(n_meta*2,n_meta)
              ]
        )

    def forward(self,input_batch):
        embedded_batch=self.embedding(input_batch)
        embedded_batch=self.positional_encoding(embedded_batch)
        encoded=self.encoder(embedded_batch)[0]
        meta=self.meta_network(encoded)

        return meta
