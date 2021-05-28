#!/usr/bin/env python
# coding=utf-8
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import pdb

def init_weights(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data,0.1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0,0.01)
            m.bias.data.zero_()

class LayerNormLSTMCell(nn.LSTMCell):
    def __init__(self, input_size, hidden_size):
        super(LayerNormLSTMCell, self).__init__(input_size, hidden_size, bias=True)

        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)

    def forward(self, x, hidden=None):
        '''
        x: [batch, input_size]
        hidden: (h_t, c_t)
        '''
        self.check_forward_input(x)
        if hidden is None:
            hx = x.new_zeros(x.size(0), self.hidden_size, requires_grad=False)
            cx = x.new_zeros(x.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden
        self.check_forward_hidden(x, hx, '[0]')
        self.check_forward_hidden(x, cx, '[1]')

        gates = self.ln_ih(F.linear(x, self.weight_ih, self.bias_ih)) \
                + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        i, f, o = gates[:, :(3 * self.hidden_size)].sigmoid().chunk(3, 1)
        g = gates[:, (3 * self.hidden_size):].tanh()

        cy = (f * cx) + (i * g)
        hy = o * self.ln_ho(cy).tanh()

        return hy, cy


class LayerNormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=False):
        super(LayerNormLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        num_directions = 2 if bidirectional else 1
        self.hidden0 = nn.ModuleList([
            LayerNormLSTMCell(input_size=input_size if i == 0 else input_size * num_directions,
                              hidden_size=hidden_size) \
            for i in range(num_layers)
        ])
        if self.bidirectional:
            self.hidden1 = nn.ModuleList([
                LayerNormLSTMCell(input_size=input_size if i == 0 else input_size * num_directions,
                                  hidden_size=hidden_size) \
                for i in range(num_layers)
            ])
        
        self.h_t = Parameter(torch.zeros((1, hidden_size), dtype=torch.float), requires_grad=False)
        self.c_t = Parameter(torch.zeros((1, hidden_size), dtype=torch.float), requires_grad=False)

    def forward(self, input):
        '''
        input: [batch, seq, input_size]

        return:
            [batch, seq, num_directions * hidden_size]
        '''
        batch, seq, _ = input.size()
        num_directions = 2 if self.bidirectional else 1
        input = input.permute(1, 0, 2)
        assert input.size() == torch.Size((seq, batch, self.input_size))

        h_0, c_0 = self.h_t.expand(batch, -1), self.c_t.expand(batch, -1)
        # [seq, num_layers, batch, num_directions * hidden_size]
        h_t = [[[None,] * batch for _ in range(self.num_layers)] for _ in range(seq)] 
        c_t = [[[None,] * batch for _ in range(self.num_layers)] for _ in range(seq)]
        
        if self.bidirectional:
            xs = input
            for l, (layer0, layer1) in enumerate(zip(self.hidden0, self.hidden1)):
                h0, c0 = h_0, c_0
                h1, c1 = h_0, c_0
                for t, (x0, x1) in enumerate(zip(xs, reversed(xs))):
                    # [batch, hidden_size]
                    h0, c0 = layer0(x0, (h0, c0))
                    h1, c1 = layer1(x1, (h1, c1))
                    # [batch, 2 * hidden_size]
                    h_t[t][l] = torch.cat([h0, h1], dim=-1)
                    c_t[t][l] = torch.cat([c0, c1], dim=-1)
                xs = [h_t[t][l] for t in range(seq)]
        else:
            xs = input
            for l, layer0 in enumerate(self.hidden0):
                h0, c0 = h_0, c_0
                for t, x0 in enumerate(xs):
                    # [batch, hidden_size]
                    h0, c0 = layer0(x0, (h0, c0))
                    # [batch, hidden_size]
                    h_t[t][l] = h0
                    c_t[t][l] = c0
                xs = [h_t[t][l] for t in range(seq)]

        # [seq, batch, num_directions * hidden_size]
        y = torch.stack([h_t[t][-1] for t in range(seq)], dim=0)
        assert y.size() == torch.Size((seq, batch, num_directions * self.hidden_size))

        return y.permute(1, 0, 2), (h_t, c_t)


class AttentionLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 1),
        )
        self.apply(init_weights)

    def forward(self, query, key, value, q_mask, k_mask):
        '''
        query: [batch, seq1, hidden_size]
        key: [batch, seq2, hidden_size * num_hidden_state]
        value: [batch, seq2, hidden_size * num_hidden_state]
        q_mask: [batch, seq1]
        k_mask: [batch, seq2]

        return:
            [batch, seq1, hidden_size]
        '''
        batch, seq1, hidden_size1 = query.size()
        _, seq2, hidden_size2 = key.size()

        mask = q_mask.unsqueeze(2).matmul(k_mask.unsqueeze(1))
        assert mask.size() == torch.Size((batch, seq1, seq2))
        
        query_e = query.unsqueeze(2).expand(-1, -1, seq2, -1)
        key_e = key.unsqueeze(1).expand(-1, seq1, -1, -1)
        stack = torch.cat([query_e, key_e], dim=-1)
        assert stack.size() == torch.Size((batch, seq1, seq2, hidden_size1 + hidden_size2))
        # [batch, seq1, seq2]
        A = self.mlp(stack) \
                .squeeze(-1) \
                .masked_fill(mask == 0, float('-inf')) \
                .exp()
        A_sum = A.sum(dim=-1, keepdim=True).clamp(min=2e-15)
        attn = A.div(A_sum)
        assert A.size() == torch.Size((batch, seq1, seq2))
        return attn.matmul(value)


class MLPLayer(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=None):
        super(MLPLayer, self).__init__()
        if hidden_size is None:
            hidden_size = in_size
        self.mlp = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, out_size),
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.mlp(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, q_mask, k_mask, scale=None):
        '''
        q: [B, L_q, D_q]
        k: [B, L_k, D_k]
        v: [B, L_v, D_v]
        q_mask: [B, L_q]
        k_mask: [B, L_k]
        '''
        batch, L_q, D_q = query.size()
        _, L_k, D_k = key.size()

        if scale is None:
            scale = D_q

        mask = q_mask.unsqueeze(2).matmul(k_mask.unsqueeze(1))
        assert mask.size() == torch.Size((batch, L_q, L_k))

        # [batch, L_q, L_k]
        A = query.matmul(key.transpose(1, 2)) \
                .div(np.sqrt(scale)) \
                .masked_fill(mask == 0, float('-inf')) \
                .exp()
        A_sum = A.sum(dim=-1, keepdim=True).clamp(min=2e-15)
        attn = A.div(A_sum)
        assert attn.size() == torch.Size((batch, L_q, L_k))
        return attn.matmul(value)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, nheads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dim_head = dim // nheads
        self.nheads = nheads
        self.linear_k = nn.Linear(dim, self.dim_head * nheads)
        self.linear_v = nn.Linear(dim, self.dim_head * nheads)
        self.linear_q = nn.Linear(dim, self.dim_head * nheads)

        self.dot_product_attn = ScaledDotProductAttention()
        self.linear_final = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(dim)
        self.apply(init_weights)

    def forward(self, query, key, value, q_mask, k_mask):
        '''
        key: [B, L_k, D_k]
        value: [B, L_v, D_v]
        query: [B, L_q, D_q]
        q_mask: [B, L_q]
        k_mask: [k_q]
        '''
        residual = query
        batch = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        key = key.view(batch * self.nheads, -1, self.dim_head)
        value = value.view(batch * self.nheads, -1, self.dim_head)
        query = query.view(batch * self.nheads, -1, self.dim_head)

        q_mask = q_mask.repeat(self.nheads, 1)
        k_mask = k_mask.repeat(self.nheads, 1)

        context = self.dot_product_attn(query=query,
                                        key=key,
                                        value=value,
                                        q_mask=q_mask,
                                        k_mask=k_mask,
                                        scale=self.dim_head)
        context = context.view(batch, -1, self.nheads * self.dim_head)
        
        output = self.linear_final(context)
        output = self.dropout(output)

        output = self.layer_norm(residual + output)

        return output

class PositionalWiseFeedForward(nn.Module):
    def __init__(self, dim=512, ffn_dim=2048, dropout=0.1):
        super(PositionalWiseFeedForward, self).__init__()
        #self.w1 = nn.Conv1d(dim, ffn_dim, 1)
        #self.w2 = nn.Conv1d(dim, ffn_dim, 1)
        self.fc = MLPLayer(in_size=dim, hidden_size=ffn_dim, out_size=dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        self.apply(init_weights)

    def forward(self, x):
        '''
        x: [B, S, D]
        '''
        #output = x.transpose(1, 2)
        #output = self.w2(torch.relu(self.w1(output)))
        #output = self.dropout(output.transpose(1, 2))
        output = self.dropout(self.fc(x))

        return self.layer_norm(x + output)

class Transformer(nn.Module):
    def __init__(self, dim, nheads=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.nheads = nheads
        self.attention = MultiHeadAttention(dim=dim, nheads=nheads, dropout=dropout)
        dim = (dim // nheads) * nheads
        self.pos_fc = PositionalWiseFeedForward(dim=dim, dropout=dropout, ffn_dim=2 * dim)
    
    #def init_weights(self):
    #    initrange = 0.1
    #    nn.init.uniform_(self.encoder.weight, -initrange, initrange)
    #    for module in self.decoder.modules():
    #        nn.init.zeros_(module.weight)
    #        nn.init.uniform_(module.weight, -initrange, initrange)
    #    if self.dueling:
    #        nn.init.zeros_(self.value_layer.weight)
    #        nn.init.uniform_(self.value_layer, -initrange, initrange)
    
    def forward(self, query, q_mask, key=None, value=None, k_mask=None):
        '''
        query: [B, L_q, D_q]
        key: [B, L_k, D_k]
        value: [B, L_v, D_v]
        q_mask: [B, L_q]
        k_mask: [B, L_k]
        '''
        if key is None:
            key = query
            value = query
            k_mask = q_mask

        B, L_q, dim = query.size()
        L_k = key.size(1)

        output = self.attention(
            query=query,
            key=key,
            value=value,
            q_mask=q_mask,
            k_mask=k_mask
        )
        dim = (dim // self.nheads) * self.nheads
        assert output.size() == torch.Size((B, L_q, dim))
        output = self.pos_fc(output)
        return output, q_mask
