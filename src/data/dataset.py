#!/usr/bin/env python
# coding=utf-8
import os
import pickle
import random
from typing import List

from torch.utils.data import Dataset

from .structure import *

class FeverDataset(Dataset):
    def __init__(self, file_name_or_path: str, label2id: dict=None):
        super(FeverDataset, self).__init__()
        self.is_dir = os.path.isdir(file_name_or_path)
        self.label2id = label2id
        if self.is_dir:
            names = sorted(os.listdir(file_name_or_path), key=lambda x: int(x[:-3]))
            self.data = list(map(lambda f: os.path.join(file_name_or_path, f), names))
        else:
            self.data = self.load_data(file_name_or_path)

    def load_data(self, filename: str):
        with open(filename, 'rb') as fr:
            data = pickle.load(fr)
        return data

    def __getitem__(self, index: int):
        if self.is_dir:
            claim, label, evidence_set, sentences = self.load_data(self.data[index])
        else:
            claim, label, evidence_set, sentences = self.data[index]
        state = [State(claim=claim,
                      label=label,
                      evidence_set=evidence_set,
                      pred_label=idx,
                      candidate=[],
                      count=0
        ) for idx in self.label2id.values()]
        actions = [[Action(sentence=sent) for sent in sentences] for _ in range(len(self.label2id))]
        return state, actions

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    batch_state, batch_actions = [], []
    for state, actions in batch:
        batch_state.extend(state)
        batch_actions.extend(actions)
    assert len(batch_state) == len(batch_actions)
    return batch_state, batch_actions

