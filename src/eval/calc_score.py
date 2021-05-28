#!/usr/bin/env python
# coding=utf-8
import torch
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from collections import defaultdict
import json
from itertools import zip_longest, groupby
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


def calc_fever2_score(predicted_fever2_list: List[dict], predicted_fever1_list: List[dict], true_file: str) \
        -> Tuple[List[dict], float, float, float, float, float]:
    ids_fever2 = set(map(lambda item: int(item['id']), predicted_fever2_list))
    predicted_fever1_dict = {item['id']: item for item in predicted_fever1_list}
    true_data = {}
    with open(true_file, 'r') as fr:
        for line in tqdm(fr.readlines()):
            instance = json.loads(line.strip())
            true_data[instance['id']] = instance
    cut_predicted_fever1_list = []
    for idx, instance in true_data.items():
        if idx not in ids_fever2:
            predicted_fever2_list.append({
                'id': instance['id'],
                'label': instance['label'],
                'evidence': instance['evidence'],
                'predicted_label': 'NOT ENOUGH INFO',
            })
        cut_predicted_fever1_list.append(predicted_fever1_dict[instance['original_id']])
    assert len(cut_predicted_fever1_list) == len(predicted_fever2_list)

    scores = {}
    strict_score, label_accuracy, precision, recall, f1 = fever_score(cut_predicted_fever1_list)
    scores['fever1'] = (strict_score, label_accuracy, precision, recall, f1)
    strict_score, label_accuracy, precision, recall, f1 = fever_score(predicted_fever2_list)
    scores['fever2'] = (strict_score, label_accuracy, precision, recall, f1)
    return predicted_fever2_list, cut_predicted_fever1_list, scores

# def truncate_q_values(predicted_state_seq: List, thred: float=0.1, is_test: bool=False):
#     def grouper(iterable, n):
#         "Collect data into fixed-length chunks or blocks"
#         # grouper('ABCDEF', 3) --> ABC DEF
#         args = [iter(iterable)] * n
#         return zip_longest(*args, fillvalue=None)
#     def all_equal(iterable):
#         "Returns True if all the elements are equal to each other"
#         g = groupby(iterable)
#         return next(g, True) and not next(g, False)
# 
#     predicted_list = []
#     for group_state_seq in grouper(predicted_state_seq, 3):
#         assert all_equal([idx for idx, _ in group_state_seq])
#         # 获取各pred_label对应的截断Q值
#         max_t = []
#         for _, state_seq in group_state_seq:
#             score_seq = [score for score, _, _, _ in state_seq]
#             score_gap = [score_seq[t] - score_seq[t - 1] for t in range(1, len(score_seq))] # cur - pre
#             ptr = len(score_seq) - 1
#             for t in range(len(score_gap) - 1, -1, -1):
#                 if score_gap[t] >= -thred and score_gap[t] <= thred:
#                     ptr = t
#                 else:
#                     break
#             max_t.append(ptr)
#         # 取截断Q值最高的那个作为预测pred_label
#         ids = torch.tensor([group_state_seq[i][1][t][0] for i, t in enumerate(max_t)]).argmax().item()
#         state_id = group_state_seq[ids][0]
#         state_seq = group_state_seq[ids][1]
#         ptr = max_t[ids]
#         predicted_list.append({
#             'id': state_id,
#             'label': state_seq[ptr][1][0],
#             'evidence': state_seq[ptr][2],
#             'predicted_label': state_seq[ptr][1][1],
#             'predicted_evidence': state_seq[ptr][3]
#         } if not is_test else {
#             'id': state_id,
#             'predicted_label': state_seq[ptr][1][1],
#             'predicted_evidence': state_seq[ptr][3]
#         })
#     return predicted_list

def truncate_q_values(predicted_state_seq: List, is_test: bool=False, is_precise: bool=False):
    def grouper(iterable, n):
        "Collect data into fixed-length chunks or blocks"
        # grouper('ABCDEF', 3) --> ABC DEF
        args = [iter(iterable)] * n
        return zip_longest(*args, fillvalue=None)
    def all_equal(iterable):
        "Returns True if all the elements are equal to each other"
        g = groupby(iterable)
        return next(g, True) and not next(g, False)

    predicted_list = []
    for group_state_seq in grouper(predicted_state_seq, 3):
        assert all_equal([idx for idx, _ in group_state_seq])
        # 获取各pred_label对应的截断Q值
        if is_precise:
            max_q_sa = float('-inf')
            ids, ptr = -1, -1
            # 取截断Q值最高的那个作为预测pred_label
            # ids 用于指向是哪个 label; ptr 用于指定是该 label 下的哪个时刻的 state
            for i, (_, state_seq) in enumerate(group_state_seq):
                pred_label = state_seq[0][1][1]
                if pred_label == 'NOT ENOUGH INFO':
                    if max_q_sa < state_seq[-1][0][0]:
                        max_q_sa = state_seq[-1][0][0]
                        ids, ptr = i, len(state_seq) - 1
                else:
                    q_sa_seq, val_s_seq = list(zip(
                        *[(q_sa, val_s) for (q_sa, val_s), _, _, _ in state_seq]))
                    q_sa_seq, val_s_seq = np.asarray(q_sa_seq), np.asarray(val_s_seq)
                    if max_q_sa < q_sa_seq.max():
                        max_q_sa = q_sa_seq.max()
                        ids, ptr = i, val_s_seq.argmax()
        else:
            ids = np.asarray([state_seq[-1][0][0] for _, state_seq in group_state_seq]).argmax()
            ptr = -1
        state_id = group_state_seq[ids][0]
        state_seq = group_state_seq[ids][1]
        predicted_list.append({
            'id': state_id,
            'label': state_seq[ptr][1][0],
            'evidence': state_seq[ptr][2],
            'predicted_label': state_seq[ptr][1][1],
            'predicted_evidence': state_seq[ptr][3]
        } if not is_test else {
            'id': state_id,
            'predicted_label': state_seq[ptr][1][1],
            'predicted_evidence': state_seq[ptr][3]
        })
    return predicted_list
