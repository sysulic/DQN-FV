#!/usr/bin/env python
# coding=utf-8
import torch
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from collections import defaultdict
import json
from itertools import zip_longest, groupby
import os
from prettytable import PrettyTable
import pdb

from .scorer import fever_score


def calc_test_result(predicted_list: List[dict], true_file: str, logger=None) -> List[dict]:
    predicted_dict = {int(item['id']): item for item in predicted_list}
    if logger:
        logger.info('Calculating test result')
        logger.info(f'Loading true data from {true_file}')
    result = []
    with open(true_file, 'r') as fr:
        for line in tqdm(fr.readlines()):
            instance = json.loads(line.strip())
            idx = int(instance['id'])
            if idx in predicted_dict:
                label = predicted_dict[idx]['predicted_label']
                evidence = predicted_dict[idx]['predicted_evidence']
            else:
                label = 'NOT ENOUGH INFO'
                evidence = []
            result.append({
                'id': idx,
                'predicted_label': label,
                'predicted_evidence': evidence
            })
    assert len(result) == 19998
    return result

def calc_fever_score(predicted_list: List[dict], true_file: str, logger=None) \
        -> Tuple[List[dict], float, float, float, float, float]:
    ids = set(map(lambda item: int(item['id']), predicted_list))
    if logger:
        logger.info('Calculating FEVER score')
        logger.info(f'Loading true data from {true_file}')
    with open(true_file, 'r') as fr:
        for line in tqdm(fr.readlines()):
            instance = json.loads(line.strip())
            if int(instance['id']) not in ids:
                predicted_list.append({
                    'id': instance['id'],
                    'label': instance['label'],
                    'evidence': instance['evidence'],
                    'predicted_label': 'NOT ENOUGH INFO',
                    'predicted_evidence': []
                })
    assert len(predicted_list) == 19998
    
    predicted_list_per_label = defaultdict(list)
    for item in predicted_list:
        predicted_list_per_label[item['label']].append(item)
    predicted_list_per_label = dict(predicted_list_per_label)

    scores = {}
    strict_score, label_accuracy, precision, recall, f1 = fever_score(predicted_list)
    scores['dev'] = (strict_score, label_accuracy, precision, recall, f1)
    if logger:
        logger.info(f'[Dev] FEVER: {strict_score}\tLA: {label_accuracy}\tACC: {precision}\tRC: {recall}\tF1: {f1}')
    for label, item in predicted_list_per_label.items():
        strict_score, label_accuracy, precision, recall, f1 = fever_score(item)
        scores[label] = (strict_score, label_accuracy, precision, recall, f1)
        if logger:
            logger.info(f'[{label}] FEVER: {strict_score}\tLA: {label_accuracy}\tACC: {precision}\tRC: {recall}\tF1: {f1}')
    return predicted_list, scores


def data_in_table(data, field_names=None, title=None):
    if field_names is None:
        field_names = ['TYPE', 'FEVER', 'LA', 'Pre', 'Recall', 'F1']
    tb = PrettyTable()
    tb.field_names = field_names
    tb.add_rows(data)
    if title is not None:
        tb.title = title
    return tb.get_string()

