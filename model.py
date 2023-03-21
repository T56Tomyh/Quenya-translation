# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:21:44 2023

@author: Jérôme
"""
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F



class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_hidden, dropout):
        super(EncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_hidden = d_hidden
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout = dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim,)
        self.fullyconnected1 = nn.Linear(embed_dim, d_hidden)
        self.relu = nn.ReLU()
        self.fullyconnected2 = nn.Linear(d_hidden, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    def forward(self, x, key_pad_mask):
        att = self.mha(x, x, x, key_padding_mask = key_pad_mask)[0]
        att = self.norm1(x+att)
        y = self.fullyconnected1(att)
        y = self.relu(y)
        y = self.fullyconnected2(y)
        return self.norm2(att+y)
                        
    
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_hidden, dropout):
        super(DecoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_hidden = d_hidden
        self.mha1 = nn.MultiheadAttention(embed_dim, num_heads, dropout = dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha2 = nn.MultiheadAttention(embed_dim, num_heads, dropout = dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fullyconnected1 = nn.Linear(embed_dim, d_hidden)
        self.relu = nn.ReLU()
        self.fullyconnected2 = nn.Linear(d_hidden, embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
    def forward(self, x, encoded, tgt_pad_mask, src_pad_mask, attn_mask):
        att = self.mha1(x, x, x, attn_mask = attn_mask, key_padding_mask = tgt_pad_mask)[0]
        att_norm = self.norm1(att + x) 
        att2 = self.mha2(att_norm, encoded, encoded, key_padding_mask = src_pad_mask)[0]
        att2_norm = self.norm2(att_norm + att2)
        y = self.fullyconnected1(att2_norm)
        y = self.relu(y)
        y = self.fullyconnected2(y)
        return self.norm3(y + att2_norm)
        
class Encoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, d_hidden, dropout):
        super(Encoder, self).__init__()
        self.layers = torch.nn.ModuleList([EncoderLayer(embed_dim, num_heads, d_hidden, dropout) for i in range(num_layers)])
    def forward(self, x, key_pad_mask):
        for layer in self.layers:
            x = layer(x, key_pad_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, d_hidden, dropout):
        super(Decoder, self).__init__()
        self.layers = torch.nn.ModuleList([DecoderLayer(embed_dim, num_heads, d_hidden, dropout) for i in range(num_layers)])
    def forward(self, x, encoded, tgt_pad_mask, src_padding_mask, attn_mask):
        for layer in self.layers:
            x= layer(x, encoded, tgt_pad_mask, src_padding_mask, attn_mask)
        return x
    
class Transformer(nn.Module):
    def __init__(self, num_layers_encoder, num_layers_decoder, embed_dim, num_heads, d_hidden, dropout, vocab_size):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers_encoder, embed_dim, num_heads, d_hidden, dropout)
        self.decoder = Decoder(num_layers_decoder, embed_dim, num_heads, d_hidden, dropout)
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.src_embedding = nn.Embedding(vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
        self.softmax_layer = nn.Softmax(dim = -1)
    def forward(self,x ,y, src_padding_mask, tgt_padding_mask, attn_mask, pos_encoding):
        x = self.src_embedding(x)
        y = self.tgt_embedding(y)
        x = x + pos_encoding
        y = y + pos_encoding
        encoded = self.encoder(x, src_padding_mask)
        decoded = self.decoder(y, encoded, tgt_padding_mask, src_padding_mask, attn_mask)
        logits = self.linear(decoded)
        output = self.softmax_layer(logits)
        return output
    

torch.set_default_tensor_type(torch.DoubleTensor)    

# model = Transformer(3,3,512,8,2048,0.1,150).to('cuda')
# a = torch.randint(150,(40,146)).to('cuda')
# a1 = torch.randint(150,(40,146)).to('cuda')
# b = torch.randint(1,(40,146)).to('cuda')
# c = torch.randint(1,(146,146)).bool().to('cuda')
# pos = torch.rand((146,512)).to('cuda')

# criterion = torch.nn.CrossEntropyLoss()
# model.train()
# num_epochs = 10
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# outputs = model(a,a1,b,b,c,pos)
# loss = criterion(outputs.view(-1,150,146), a1)

#         # Backward pass
# loss.backward()
#         # Update the weights
# optimizer.step()
