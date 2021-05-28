#!/usr/bin/env python3
# coding=utf-8
from tqdm import tqdm, trange
from functools import reduce
from typing import List, Tuple
from copy import deepcopy
import os
import json
import pickle
import pdb

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence
from torch.optim import SGD, Adam, AdamW
#from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
#from torch.utils.data.distributed import DistributedSampler

from .base_dqn import BaseDQN
from .lstm_dqn import lstm_load_and_process_data
from data.structure import *
from data.dataset import FeverDataset


ggnn_load_and_process_data = lstm_load_and_process_data

def convert_tensor_to_ggnn_inputs(batch_state_tensor: List[torch.Tensor],
                                  batch_actions_tensor: List[torch.Tensor],
                                  device=None) -> dict:
    device = device if device != None else torch.device('cpu')

    state_len = [state.size(0) for state in batch_state_tensor]
    actions_len = [action.size(0) for action in batch_actions_tensor]
    s_max, a_max = max(state_len), max(actions_len)

    state_pad = pad_sequence(batch_state_tensor, batch_first=True)
    actions_pad = pad_sequence(batch_actions_tensor, batch_first=True)

    state_mask = torch.tensor([[1] * size + [0] * (s_max - size) for size in state_len],
                             dtype=torch.float)
    actions_mask = torch.tensor([[1] * size + [0] * (a_max - size) for size in actions_len],
                                dtype=torch.float)
    adj_matrix = torch.tensor([[[1. / size] * size + [0] * (s_max - size) \
                                if i < size else [0] * s_max \
                                for i in range(s_max)] for size in state_len],
                              dtype=torch.float)
    return {
        'states': state_pad.to(device),
        'actions': actions_pad.to(device),
        'state_mask': state_mask.to(device),
        'actions_mask': actions_mask.to(device),
        'adj_matrix': adj_matrix.to(device)
    }


class QNetwork(nn.Module):
    def __init__(self,
                 input_size,
                 num_labels,
                 hidden_size=None,
                 dropout=0.1,
                 n_steps=3,
                 dueling=True,
                 aggregate='attn_mean'):
        super(QNetwork, self).__init__()
        if hidden_size is None:
            hidden_size = input_size
        self.n_steps = n_steps
        self.aggregate = aggregate
        self.dueling = dueling
        
        #GGNN
        self.reset_gate = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.update_gate = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.transform = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.node_fc = nn.Linear(hidden_size, hidden_size)

        if self.aggregate.find('attn') != -1:
            # attention paramters
            self.attn_layer = nn.Sequential(
                nn.Linear(
                    hidden_size * 2,
                    hidden_size
                ),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
                nn.ReLU(True)
            )
        # Value
        if dueling:
            self.value_layer = nn.Linear(hidden_size, 1)
        # Advantage
        self.weight = Parameter(torch.Tensor(num_labels,
                                             hidden_size,
                                             hidden_size))
        self.bias = Parameter(torch.Tensor(num_labels))
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            feature_in = self.weight.size(2)
            bound = 1 / np.sqrt(feature_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def ggnn(self, nodes, adj_matrix):
        '''
        nodes: [batch, seq, hidden_size]
        adj_matrix: [batch, seq, seq]
        '''
        hidden_state = nodes
        for step in range(self.n_steps):
            prop_hidden_state = self.node_fc(hidden_state)
            # propagate
            a = adj_matrix.matmul(torch.cat((prop_hidden_state, hidden_state), dim=2))
            # update
            z = self.update_gate(a)
            r = self.reset_gate(a)
            h_hat = self.transform(torch.cat((prop_hidden_state, r * hidden_state), dim=2))
            hidden_state = (1 - z) * hidden_state + z * h_hat
        return hidden_state

    def attention_aggregate(self, query, key, value, q_mask, k_mask):
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
        A = self.attn_layer(stack) \
                .squeeze(-1) \
                .masked_fill(torch.logical_not(mask), float('-inf')) \
                .exp()
        A_sum = A.sum(dim=-1, keepdim=True).clamp(min=2e-15)
        attn = A.div(A_sum)
        assert A.size() == torch.Size((batch, seq1, seq2))
        return attn.matmul(value)

    def calc_state_semantic(self, features, mask, mode):
        batch, _, hidden_size = features.size()
        if mode == 'mean':
            num = mask.sum(dim=1).view(-1, 1, 1).expand(-1, 1, hidden_size)
            state_semantic = features.sum(dim=1, keepdim=True).div(num)
            assert num.size() == torch.Size((batch, 1, hidden_size))
        elif mode == 'max':
            state_semantic = features \
                                .masked_fill(mask.unsqueeze(2) == 0,
                                             float('-inf')) \
                                .max(dim=1, keepdim=True)[0]
        else:
            raise ValueError('mode error')
        return state_semantic

    def forward(self, states, state_mask, actions, actions_mask, adj_matrix):
        '''
        states: [batch, seq, hidden_size]
        state_mask: [batch, seq]
        actions: [batch, seq2, hidden_size]
        actions_mask: [batch, seq2]
        adj_matrix: [batch, seq, seq]

        return:
            [batch * available_seq2, num_labels]
        '''
        batch, seq, hidden_size = states.size()
        seq2, num_labels = actions.size(1), 3
        # GGNN
        out = self.ggnn(states, adj_matrix)
        assert out.size() == torch.Size((batch, seq, hidden_size))

        if self.aggregate.find('attn') != -1:
            # [batch, seq2, hidden_size * num_hidden_state]
            states_feat = self.attention_aggregate(actions,
                                                   out, out,
                                                   actions_mask,
                                                   state_mask)
            state_semantic = self.calc_state_semantic(states_feat, actions_mask,
                                                      'mean' if self.aggregate.find('mean') != -1 else 'max')
        else:
            if self.aggregate in {'max', 'mean'}:
                state_semantic = self.calc_state_semantic(out, state_mask, self.aggregate)
            elif self.aggregate == 'last':
                last_step = state_mask.sum(dim=1) \
                                .sub(1) \
                                .type(torch.long) \
                                .view(-1, 1, 1) \
                                .expand(-1, -1, 2 * hidden_size)
                state_semantic = out.gather(dim=1, index=last_step)
            states_feat = state_semantic.expand(-1, seq2, -1)
        assert state_semantic.size() == torch.Size((batch, 1, hidden_size))
        assert states_feat.size() == torch.Size((batch, seq2, hidden_size))
        
        # [batch, num_labels, hidden_size, seq2]
        ws = self.weight.unsqueeze(0).matmul(states_feat.unsqueeze(1).transpose(3, 2))
        assert ws.size() == torch.Size((batch, num_labels, hidden_size, seq2))

        # Value - [batch, seq2, num_labels]
        if self.dueling:
            val_scores = self.value_layer(state_semantic)
            assert val_scores.size() == torch.Size((batch, 1, 1))

            val_scores = val_scores.expand(-1, seq2, num_labels)
            assert val_scores.size() == torch.Size((batch, seq2, num_labels))
        
        # Advantage - [batch, seq2, num_labels]
        adv_scores = actions.transpose(2, 1).unsqueeze(1).mul(ws).sum(dim=2) + self.bias[None,:,None]
        adv_scores = adv_scores.permute(0, 2, 1)
        assert adv_scores.size() == torch.Size((batch, seq2, num_labels))
        
        # Q value - [batch, seq2, num_labels]
        if self.dueling:
            q_value = val_scores + adv_scores - adv_scores.mean(dim=(2, 1), keepdim=True)
        else:
            q_value = adv_scores
        assert q_value.size() == torch.Size((batch, seq2, num_labels))
        
        # 去除padding的action对应的score
        no_pad = actions_mask.view(-1).nonzero().view(-1)
        q_value = q_value.reshape(-1, num_labels)[no_pad]
        return (q_value,)

class GGNNDQN(BaseDQN):
    def __init__(self, args):
        super(GGNNDQN, self).__init__(args)
        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model_type = args.model_name_or_path.lower().split('/')[-1]
        HIDDEN_SIZE = {
            'bert-base-uncased': 768,
            'bert-base-cased': 768,
            'albert-large-v2': 1024,
            'gear-pretrained': 768
        }
        # q network
        self.q_net = QNetwork(
            input_size=HIDDEN_SIZE[model_type],
            hidden_size=HIDDEN_SIZE[model_type],
            num_labels=args.num_labels,
            dueling=args.dueling,
            aggregate=args.aggregate,
        )
        # Target network
        self.t_net = deepcopy(self.q_net) if args.do_train else self.q_net
        self.q_net.zero_grad()

        self.set_network_untrainable(self.t_net)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        
        #self.optimizer = SGD(self.q_net.parameters(), lr=args.learning_rate, momentum=0.9)
        self.optimizer = AdamW(self.q_net.parameters(), lr=args.learning_rate)
        #self.optimizer = Adam(self.q_net.parameters(), lr=args.learning_rate)



    def convert_to_inputs_for_select_action(self, batch_state: List[State], batch_actions: List[List[Action]]) -> List[dict]:
        assert len(batch_state) == len(batch_actions)
        batch_state_tensor, batch_actions_tensor = [], []
        for state, actions in zip(batch_state, batch_actions):
            # tokens here is actually the sentence embedding
            ## [seq, dim]
            state_tensor = torch.tensor([state.claim.tokens] + [sent.tokens for sent in state.candidate],
                                        dtype=torch.float)
            ## [seq2, dim]
            actions_tensor = torch.tensor([action.sentence.tokens for action in actions],
                                          dtype=torch.float)
            batch_state_tensor.append(state_tensor)
            batch_actions_tensor.append(actions_tensor)
        return convert_tensor_to_ggnn_inputs(batch_state_tensor, batch_actions_tensor)
    

    def convert_to_inputs_for_update(self, states: List[State], actions: List[Action]) -> dict:
        assert len(states) == len(actions)
        batch_state_tensor, batch_actions_tensor = [], []
        for state, action in zip(states, actions):
            ## [seq, dim]
            state_tensor = torch.tensor([state.claim.tokens] + [sent.tokens for sent in state.candidate],
                                        dtype=torch.float)
            ## [seq2, dim]
            actions_tensor = torch.tensor([action.sentence.tokens], dtype=torch.float)
            batch_state_tensor.append(state_tensor)
            batch_actions_tensor.append(actions_tensor)
        return convert_tensor_to_ggnn_inputs(batch_state_tensor, batch_actions_tensor, self.device)

