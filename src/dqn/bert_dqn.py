#!/usr/bin/env python3
# coding=utf-8
from tqdm import tqdm, trange
from functools import reduce
from typing import List, Tuple
import pdb
import os
import json
import pickle

import numpy as np
import torch
from torch.optim import SGD
#from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
#from torch.utils.data.distributed import DistributedSampler
from transformers import (
    WEIGHTS_NAME,
    #AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

from .base_dqn import BaseDQN
from data.structure import *
from data.dataset import FeverDataset

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            XLMRobertaConfig,
            FlaubertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}

def bert_load_and_process_data(args: dict, filename: str, token_fn: 'function') \
        -> DataSet:
    if filename.find('train') != -1:
        mode = 'train'
    elif filename.find('dev') != -1:
        mode = 'dev'
    else:
        mode = 'test'

    cached_file = os.path.join(
        '/'.join(filename.split('/')[:-1]),
        'cached_{}_{}_{}.pk'.format(mode,
                                    list(filter(None, args.model_name_or_path.split('/'))).pop(),
                                    args.max_sent_length)
    )
    data = None
    if not os.path.exists(cached_file):
        args.logger.info(f'Loading and processing data from {filename}')
        data = []
        with open(filename, 'rb') as fr:
            for line in tqdm(fr.readlines()):
                instance = json.loads(line.decode('utf-8').strip())
                claim = Claim(id=instance['id'],
                              str=instance['claim'],
                              tokens=token_fn(instance['claim'], max_length=args.max_sent_length))
                sent2id = {}
                sentences = []
                for title, text in instance['documents'].items():
                    for line_num, sentence in text.items():
                        sentences.append(Sentence(id=(title, int(line_num)),
                                                  str=sentence,
                                                  tokens=token_fn(sentence, max_length=args.max_sent_length)))
                        sent2id[(title, int(line_num))] = len(sentences) - 1
                if mode == 'train':
                    label = args.label2id[instance['label']]
                    evidence_set = [[sentences[sent2id[(title, int(line_num))]] \
                                        for title, line_num in evi] \
                                            for evi in instance['evidence_set']]
                elif mode == 'dev':
                    label = args.label2id[instance['label']]
                    evidence_set = instance['evidence_set']
                else:
                    label = None
                    evidence_set = None
                data.append((claim, label, evidence_set, sentences))
            with open(cached_file, 'wb') as fw:
                pickle.dump(data, fw)
    dataset = FeverDataset(cached_file, label2id=args.label2id)
    return dataset


def convert_tokens_to_bert_inputs(all_tokens_a: List[int],
                                  all_tokens_b: List[int],
                                  max_seq_len: int, 
                                  CLS: int, SEP: int, device=None) -> dict:
    device = torch.device('cpu') if device is None else device
    
    all_inputs_ids = []
    all_inputs_mask = []
    all_segment_ids = []
    for tokens_a, tokens_b in zip(all_tokens_a, all_tokens_b):
        b_len = max_seq_len - 3 - len(tokens_a)
        assert b_len > 0
        inputs_ids = (CLS,) + tokens_a + (SEP,) + tokens_b[:b_len] + (SEP,)
        inputs_mask = (1,) * len(inputs_ids)
        segment_ids = (0,) * (2 + len(tokens_a)) + (1,) * (len(inputs_ids) - len(tokens_a) - 2)
        assert len(inputs_ids) == len(inputs_mask) == len(segment_ids)
        
        padding = (0,) * (max_seq_len - len(inputs_ids))
        all_inputs_ids.append(inputs_ids + padding)
        all_inputs_mask.append(inputs_mask + padding)
        all_segment_ids.append(segment_ids + padding)

    return {
        'input_ids': torch.tensor(all_inputs_ids, dtype=torch.long, device=device),
        'attention_mask': torch.tensor(all_inputs_mask, dtype=torch.long, device=device),
        'token_type_ids': torch.tensor(all_segment_ids, dtype=torch.long, device=device)
    }


class BertDQN(BaseDQN):
    def __init__(self, args):
        super(BertDQN, self).__init__(args)
        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(
            args.model_name_or_path,
            num_labels=args.num_labels,
            finetuning_task=args.task_name,
            #cache_dir=args.cache_dir if argsche_dir else None,
        )
        self.tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            #cache_dir=args.cache_dir if args.cache_dir else None,
        )
        # Q network
        self.q_net = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            #cache_dir=args.cache_dir if args.cache_dir else None,
        )
        # Target network
        self.t_net = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            #cache_dir=args.cache_dir if args.cache_dir else None,
        ) if args.do_train else self.q_net
        self.q_net.zero_grad()
        self.set_network_untrainable(self.t_net)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        
        ## Prepare optimizer and schedule (linear warmup and decay)
        #no_decay = ["bias", "LayerNorm.weight"]
        #optimizer_grouped_parameters = [
        #    {
        #        "params": [p for n, p in self.q_net.named_parameters() if not any(nd in n for nd in no_decay)],
        #        "weight_decay": args.weight_decay,
        #    },
        #    {"params": [p for n, p in self.q_net.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        #]

        #self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        ##scheduler = get_linear_schedule_with_warmup(
        ##    self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        ##)
        encoder = getattr(self.q_net, args.model_type)
        classifier = getattr(self.q_net, 'classifier')
        self.optimizer = SGD([
            {'params': encoder.parameters()},
            {'params': classifier.parameters(), 'lr': args.learning_rate * 10}
        ], lr=args.learning_rate, momentum=0.9)


    def token(self, text_sequence: str, max_length: int=None) -> Tuple[int]:
        return tuple(self.tokenizer.encode(text_sequence,
                                           add_special_tokens=False,
                                           max_length=max_length))
    

    def convert_to_inputs_for_select_action(self, batch_state: List[State], batch_actions: List[List[Action]]) -> List[dict]:
        assert len(batch_state) == len(batch_actions)
        batch_tokens_a, batch_tokens_b = [], []
        max_seq_len = 0
        for state, actions in zip(batch_state, batch_actions):
            candidate = reduce(lambda seq1, seq2: seq1 + seq2,
                               [sent.tokens for sent in state.candidate]) if len(state.candidate) else ()
            length = self.max_seq_length - 3 - len(state.claim.tokens) - len(candidate)
            if length <= 0:
                self.logger.info(state.candidate)
                self.logger.info(f'claim: {len(state.claim.tokens)}; candidate: {len(candidate)}; length: {length}')
            assert length > 0
            all_tokens_a = [state.claim.tokens] * len(actions)
            all_tokens_b = [candidate + action.sentence.tokens[:length] \
                              if action.sentence != None else candidate \
                                for action in actions]
            
            batch_tokens_a.extend(all_tokens_a)
            batch_tokens_b.extend(all_tokens_b)

            cur_max_seq_len = max([len(tokens) for tokens in all_tokens_b]) \
                              + len(state.claim.tokens) + 3
            max_seq_len = max(cur_max_seq_len, max_seq_len)

        CLS, SEP = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        
        inputs = convert_tokens_to_bert_inputs(batch_tokens_a,
                                               batch_tokens_b,
                                               max_seq_len, CLS, SEP)
                                                #self.max_seq_length, CLS, SEP)
        return inputs
    

    def convert_to_inputs_for_update(self, states: List[State], actions: List[Action]) -> dict:
        assert len(states) == len(actions)
        all_tokens_a, all_tokens_b = [], []
        for state, action in zip(states, actions):
            tokens_a = state.claim.tokens
            candidate = reduce(lambda seq1, seq2: seq1 + seq2,
                               [sent.tokens for sent in state.candidate]) \
                            if len(state.candidate) else ()
            # action=None: state is terminal state
            # action.sentence=None: state is non terminal state but has no action sentence
            tokens_b = candidate
            if action is not None and action.sentence is not None:
                tokens_b = tokens_b + action.sentence.tokens
            assert len(tokens_b)
            all_tokens_a.append(tokens_a)
            all_tokens_b.append(tokens_b)
        max_seq_len = min(max([len(tokens_a) + len(tokens_b) + 3 \
                                for tokens_a, tokens_b in zip(all_tokens_a, all_tokens_b)]),
                          self.max_seq_length)
        
        CLS, SEP = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        
        return convert_tokens_to_bert_inputs(all_tokens_a, all_tokens_b,
                                             max_seq_len, CLS, SEP, self.device)
                                             #self.max_seq_length, CLS, SEP, self.device)

