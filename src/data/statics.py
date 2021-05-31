#!/usr/bin/env python
# coding=utf-8
from .structure import *
from collections import Counter, defaultdict
from typing import Tuple, List
import numpy as np
import pickle
import json
from pprint import pprint

DataList = List[Tuple[Claim, int, EvidenceSet, List[Sentence]]]

def load_data(filename: str) -> DataList:
    print(f'Loading data from {filename}')
    with open(filename, 'rb') as fr:
        data = pickle.load(fr)
    return data

def dis_statics(low, high, counter):
    interval = (high - low) / 10
    interval = np.arange(low, high + interval, interval)
    dis = {i: [0, 0] for i in range(10)}
    for k, v in counter.items():
        flag = False
        for i, (s, e) in enumerate(zip(interval[:-1], interval[1:])):
            if s <= k and k <= e:
                dis[i][0] += v
                flag = True
                break
        assert flag
    _sum_ = sum([i[0] for i in dis.values()])
    for v in dis.values():
        v[1] = float(v[0]) / _sum_
    print(interval)
    pprint(dis)


def statics(data: DataList) -> None:
    sentences_size = []
    tokens_size = []
    evi_sentences_size = []
    evi_tokens_size = []
    for _, label, evidence_set, sentences in data:
        if len(sentences) == 0:
            print(f'label: {label}; evidence_set: {evidence_set}')
            continue
        sentences_size.append(len(sentences))
        tokens_size.extend([len(sent.tokens) for sent in sentences])
        evi_sentences_size.extend([len(evidence) for evidence in evidence_set])
        evi_tokens_size.extend([sum([len(sent.tokens) for sent in evidence])  for evidence in evidence_set])
    sents_count = Counter(sentences_size)
    tokens_count = Counter(tokens_size)
    sents_np = np.asarray(sentences_size)
    tokens_np = np.asarray(tokens_size)
    print(f'sentences: {sents_np.min()}-{sents_np.max()}, {sents_np.mean()}-{sents_np.std()}')
    print(f'tokens: {tokens_np.min()}-{tokens_np.max()}, {tokens_np.mean()}-{tokens_np.std()}')
    print(f'sents_count: {sents_count[sents_np.min()]}-{sents_count[sents_np.max()]}')
    print(f'tokens_count: {tokens_count[tokens_np.min()]}-{tokens_count[tokens_np.max()]}')
    
    evi_sents_count = Counter(evi_sentences_size)
    evi_tokens_count = Counter(evi_tokens_size)
    evi_sents_np = np.asarray(evi_sentences_size)
    evi_tokens_np = np.asarray(evi_tokens_size)
    print(f'evi_sentences: {evi_sents_np.min()}-{evi_sents_np.max()}, {evi_sents_np.mean()}-{evi_sents_np.std()}')
    print(f'evi_tokens: {evi_tokens_np.min()}-{evi_tokens_np.max()}, {evi_tokens_np.mean()}-{evi_tokens_np.std()}')
    print(f'evi_sents_count: {evi_sents_count[evi_sents_np.min()]}-{evi_sents_count[evi_sents_np.max()]}')
    print(f'evi_tokens_count: {evi_tokens_count[evi_tokens_np.min()]}-{evi_tokens_count[evi_tokens_np.max()]}')
    #pprint(sents_count)
    #pprint(tokens_count)
    with open('/home/chenhch8/code_for_fever/dqn_data_statics.json', 'w') as fw:
        json.dump({
            'sentences_size': sentences_size,
            'tokens_size': tokens_size,
            'evi_sentences_size': evi_sentences_size,
            'evi_tokens_size': evi_tokens_size
        }, fw)

    print('sents')
    dis_statics(sents_np.min(), sents_np.max(), sents_count)
    print('tokens')
    dis_statics(tokens_np.min(), tokens_np.max(), tokens_count)
    print('evi_sents')
    dis_statics(evi_sents_np.min(), evi_sents_np.max(), evi_sents_count)
    print('evi_tokens')
    dis_statics(evi_tokens_np.min(), evi_tokens_np.max(), evi_tokens_count)

if __name__ == '__main__':
    statics(load_data('./data/dqn/cached_64_train.pk'))
