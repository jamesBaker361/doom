from torch.nn import RNN,GRU,LSTM,Linear,LayerNorm,LeakyReLU,BatchNorm1d
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

        self.embedding=nn.Embedding(vocab_size+1,embedding_dim)
        self.sequence_rnn=RNN(embedding_dim,hidden_size,num_layers,batch_first=True)
        self.meta_network=torch.nn.Sequential(
            *[Linear(hidden_size,hidden_size//2),
              BatchNorm1d(hidden_size//2),
              LeakyReLU(),
              Linear(hidden_size//2, n_meta*2),
              BatchNorm1d(n_meta*2),
              LeakyReLU(),
              Linear(n_meta*2,n_meta)
              ]
        )


    def forward(self,input_batch,*args,**kwargs):
        #input_batch (B,L)
        embedded_batch=self.embedding(input_batch)
        rnn_output,rnn_h=self.sequence_rnn(embedded_batch) #(B,L,H_in) #(num_layers, B,hidden_size)

        hidden=rnn_h[0].squeeze(0) #(B, hidden_size)
        meta=self.meta_network(hidden)

        return meta
    
    def to_config(self):
        return {
            "embedding_dim":self.embedding_dim,
            "vocab_size":self.embedding.num_embeddings,
            "hidden_size":self.hidden_size,
            "num_layers":self.num_layers,
            "n_meta":self.n_meta
        }
    
class ConcatRNN(torch.nn.Module):
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

        self.embedding=nn.Embedding(vocab_size+1,embedding_dim)
        self.sequence_rnn=RNN(embedding_dim,hidden_size,num_layers,batch_first=True)
        self.meta_network=torch.nn.ModuleList(
            [Linear(hidden_size+ n_meta,hidden_size//2),
              BatchNorm1d(hidden_size//2),
              LeakyReLU(),
              Linear(hidden_size//2 + n_meta, n_meta*2),
              BatchNorm1d(n_meta*2),
              LeakyReLU(),
              Linear(n_meta*2 +n_meta,n_meta)
              ]
        )


    def forward(self,input_batch,prior_values,*args,**kwargs):
        #input_batch (B,L)
        embedded_batch=self.embedding(input_batch)
        rnn_output,rnn_h=self.sequence_rnn(embedded_batch) #(B,L,H_in) #(num_layers, B,hidden_size)

        hidden=rnn_h[0].squeeze(0) #(B, hidden_size)
        meta=hidden
        for layer in self.meta_network:
            if type(layer)==Linear:
                meta=torch.stack([meta,prior_values],dim=1)
            meta=layer(meta)

        return meta
    
    def to_config(self):
        return {
            "embedding_dim":self.embedding_dim,
            "vocab_size":self.embedding.num_embeddings,
            "hidden_size":self.hidden_size,
            "num_layers":self.num_layers,
            "n_meta":self.n_meta
        }


class BasicLSTM(torch.nn.Module):
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

        self.embedding=nn.Embedding(vocab_size+1,embedding_dim)
        self.sequence_lstm=LSTM(embedding_dim,hidden_size,num_layers,batch_first=True)
        self.meta_network=torch.nn.Sequential(
            *[Linear(hidden_size,hidden_size//2),
              BatchNorm1d(hidden_size//2),
              LeakyReLU(),
              Linear(hidden_size//2, n_meta*2),
              BatchNorm1d(n_meta*2),
              LeakyReLU(),
              Linear(n_meta*2,n_meta)
              ]
        )


    def forward(self,input_batch,*args,**kwargs):
        #input_batch (B,L)
        embedded_batch=self.embedding(input_batch)
        lstm_output,(lstm_h,lstm_c)=self.sequence_lstm(embedded_batch) #(B,L,H_in) #(num_layers, B,hidden_size)

        hidden=lstm_h[0].squeeze(0) #(B, hidden_size)
        meta=self.meta_network(hidden)

        return meta
    
    def to_config(self):
        return {
            "embedding_dim":self.embedding_dim,
            "vocab_size":self.embedding.num_embeddings,
            "hidden_size":self.hidden_size,
            "num_layers":self.num_layers,
            "n_meta":self.n_meta
        }
    
class BasicGRU(torch.nn.Module):
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

        self.embedding=nn.Embedding(vocab_size+1,embedding_dim)
        self.sequence_gru=GRU(embedding_dim,hidden_size,num_layers,batch_first=True)
        self.meta_network=torch.nn.Sequential(
            *[Linear(hidden_size,hidden_size//2),
              BatchNorm1d(hidden_size//2),
              LeakyReLU(),
              Linear(hidden_size//2, n_meta*2),
              BatchNorm1d(n_meta*2),
              LeakyReLU(),
              Linear(n_meta*2,n_meta)
              ]
        )


    def forward(self,input_batch,*args,**kwargs):
        #input_batch (B,L)
        embedded_batch=self.embedding(input_batch)
        gru_output,gru_h=self.sequence_gru(embedded_batch) #(B,L,H_in) #(num_layers, B,hidden_size)

        hidden=gru_h[0].squeeze(0) #(B, hidden_size)
        meta=self.meta_network(hidden)

        return meta
    
    def to_config(self):
        return {
            "embedding_dim":self.embedding_dim,
            "vocab_size":self.embedding.num_embeddings,
            "hidden_size":self.hidden_size,
            "num_layers":self.num_layers,
            "n_meta":self.n_meta
        }
        

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
            x: Tensor, shape ``[, batch_size, seq_len, embedding_dim]``
        """
        x=x.view((x.size(1),x.size(0),-1))
        x = x + self.pe[:x.size(0)]
        x= self.dropout(x)
        #x=x.view((x.size(1),x.size(0),-1))
        return x
    

class BasicTransformer(torch.nn.Module):
    def __init__(self, embedding_dim,vocab_size,nhead,num_layers,n_meta):
        super().__init__()
        self.positional_encoding=PositionalEncoding(embedding_dim)
        self.embedding=nn.Embedding(vocab_size+1,embedding_dim)
        encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,nhead=nhead)
        self.encoder=nn.TransformerEncoder(encoder_layer,num_layers)
        self.nhead=nhead
        self.num_layers=num_layers
        self.n_meta=n_meta

        self.meta_network=torch.nn.Sequential(
            *[Linear(embedding_dim,embedding_dim//2),
              BatchNorm1d(embedding_dim//2),
              LeakyReLU(),
              Linear(embedding_dim//2, n_meta*2),
              BatchNorm1d(n_meta*2),
              LeakyReLU(),
              Linear(n_meta*2,n_meta)
              ]
        )

    def forward(self,input_batch,*args,**kwargs):
        embedded_batch=self.embedding(input_batch)
        embedded_batch=self.positional_encoding(embedded_batch)
        encoded=self.encoder(embedded_batch)[0]
        meta=self.meta_network(encoded)

        return meta

    def to_config(self):
        return {
            "embedding_dim":self.embedding.embedding_dim,
            "vocab_size":self.embedding.num_embeddings,
            "num_layers":self.num_layers,
            "n_meta":self.n_meta,
            "nhead":self.nhead
        }
    
class BasicCNN(torch.nn.Module):
    def __init__(self,embedding_dim,vocab_size,num_layers,n_meta):
        super().__init__()
        self.positional_encoding=PositionalEncoding(embedding_dim)
        self.embedding=nn.Embedding(vocab_size+1,embedding_dim)
        self.num_layers=num_layers
        self.n_meta=n_meta
        meta_layer_list=[]
        dim=embedding_dim
        for _ in range(num_layers):
            meta_layer_list+=[nn.Conv1d(dim,dim//2,4,2),BatchNorm1d(dim//2),LeakyReLU()]
            dim=dim//2

        meta_layer_list.append(nn.Flatten())
        meta_layer_list.append(nn.Linear(dim,n_meta))

        self.meta_network=nn.Sequential(*meta_layer_list)

    def __call__(self, input_batch,*args,**kwargs):
        print("before",input_batch.size())
        input_batch=self.embedding(input_batch)
        input_batch_size=input_batch.size()
        B=input_batch_size[0]
        input_batch=input_batch.view((B, input_batch_size[-1],input_batch_size[-2]))
        print("after",input_batch.size())
        input_batch= self.meta_network(input_batch)
        input_batch_size=input_batch.size()
        B=input_batch_size[0]
        input_batch=input_batch.view((B, input_batch_size[-1],input_batch_size[-2]))
        print("after",input_batch.size())
        return input_batch
    