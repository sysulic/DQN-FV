#!/usr/bin/env python3
# coding=utf-8
from tqdm import tqdm, trange
from functools import reduce
from copy import deepcopy
import pdb
import math
from typing import Tuple, List

import pdb

import numpy as np
import torch
from torch import nn
#from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence
from torch.optim import SGD, Adam, AdamW, lr_scheduler
#from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
#from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AlbertConfig,
    BertConfig,
    XLNetConfig,
    RobertaConfig,
)

from .base_dqn import BaseDQN
from data.structure import *
from networks import Transformer, MLPLayer, AttentionLayer

CONFIG_CLASSES = {
    'bert': BertConfig,
    'albert': AlbertConfig,
    'xlnet': XLNetConfig,
    'roberta': RobertaConfig
}


def convert_tensor_to_transformer_inputs(batch_claims: List[List[float]],
                                         batch_evidences: List[torch.Tensor],
                                         device=None) -> dict:
    device = device if device != None else torch.device('cpu')
    
    evi_len = [evi.size(0) for evi in batch_evidences]
    evi_max = max(evi_len)

    claims_tensor = torch.tensor(batch_claims, device=device)

    evidences_pad = pad_sequence(batch_evidences, batch_first=True).to(device)

    evidences_mask = torch.tensor(
        [[1] * size + [0] * (evi_max - size) for size in evi_len],
        dtype=torch.float,
        device=device
    )

    return {
        'claims': claims_tensor,
        'evidences': evidences_pad,
        'evidences_mask': evidences_mask
    }


class QNetwork(nn.Module):
    def __init__(self,
                 num_labels,
                 hidden_size,
                 dropout=0.1,
                 nheads=8,
                 num_layers=3,
                 aggregate='transformer'):
        super(QNetwork, self).__init__()
        self.nheads = nheads
        self.num_layers = num_layers
        self.fc = nn.Linear(hidden_size, hidden_size, bias=True)
        # evidence module
        for i in range(num_layers):
            setattr(self, 'evi_%d' % i, Transformer(
                dim=hidden_size,
                nheads=nheads,
                dropout=dropout
            ))
        # claim-evidence module
        self.aggregate = aggregate
        if aggregate == 'transformer':
            self.clm_evi = Transformer(dim=hidden_size,
                                       nheads=nheads,
                                       dropout=dropout)
        else:
            self.clm_evi = AttentionLayer(
                input_size=2 * hidden_size,
                hidden_size=hidden_size
            )
        # value module
        self.val = MLPLayer(in_size=2 * hidden_size,
                            hidden_size=hidden_size,
                            out_size=num_labels)

    def forward(self, claims, evidences, evidences_mask):
        '''
        claims: [batch, hidden_size]
        evidences: [batch, seq, hidden_size]
        evidences_mask: [batch, seq]

        return:
            [batch, num_labels]
        '''
        batch, seq, hidden_size = evidences.size()
        num_labels, nheads = 3, self.nheads
        
        for i in range(self.num_layers):
            layer = getattr(self, 'evi_%d' % i)
            evidences, _ = layer(
                query=evidences,
                key=evidences,
                value=evidences,
                q_mask=evidences_mask,
                k_mask=evidences_mask
            )
        assert evidences.size() == torch.Size((batch, seq, hidden_size))
        output = self.clm_evi(
            query = claims.unsqueeze(1),
            key=evidences,
            value=evidences,
            q_mask=torch.ones([batch, 1], dtype=torch.float).to(evidences_mask),
            k_mask=evidences_mask
        )
        if self.aggregate == 'transformer':
            output = output[0]
        assert output.size() == torch.Size((batch, 1, hidden_size))

        claims = self.fc(claims)
        assert claims.size() == torch.Size((batch, hidden_size))
        q_value = self.val(torch.cat([claims, output.squeeze(1)], dim=-1))
        assert q_value.size() == torch.Size((batch, 3))
        #pdb.set_trace()

        return (q_value,)


class TransformerDQN(BaseDQN):
    def __init__(self, args):
        super(TransformerDQN, self).__init__(args)
        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        config_class = CONFIG_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.model_name_or_path)
        # q network
        self.q_net = QNetwork(
            hidden_size=config.hidden_size,
            #dropout=config.dropout,
            num_labels=args.num_labels,
            nheads=args.nhead,
            num_layers=args.num_layers,
            aggregate=args.aggregate
        )
        self.logger.info(self.q_net)
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
        batch_claims, batch_evidences = [], []
        for state, actions in zip(batch_state, batch_actions):
            # tokens here is actually the sentence embedding
            ## [1, dim]
            batch_claims.extend([state.claim.tokens] * len(actions))
            ## [seq, dim]
            evidence = [sent.tokens for sent in state.candidate]
            batch_evidences.extend(
                [torch.tensor(evidence + [action.sentence.tokens],
                              dtype=torch.float) for action in actions],
            )
        return convert_tensor_to_transformer_inputs(batch_claims, batch_evidences)
    

    def convert_to_inputs_for_update(self, states: List[State], actions: List[Action]) -> dict:
        assert len(states) == len(actions)
        batch_claims, batch_evidences = [], []
        for state, action in zip(states, actions):
            ## [1, dim]
            batch_claims.append(state.claim.tokens)
            ## [seq, dim]
            evidence = [sent.tokens for sent in state.candidate]
            batch_evidences.append(
                torch.tensor(evidence + [action.sentence.tokens],
                             dtype=torch.float)
            )
        return convert_tensor_to_transformer_inputs(batch_claims, batch_evidences, self.device)

