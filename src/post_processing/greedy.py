#!/usr/bin/env python
# coding=utf-8
import numpy as np

def imprecise(data, is_test=False):
    result = []
    for idx, label, evidence_set, pred_result in data:
        label_list, evi_list, score_list = [], [], []
        for pred_label, score_seq, evi_seq in pred_result:
            label_list.append(pred_label)
            score_list.append(score_seq[-1])
            evi_list.append(evi_seq[-1])
        target_id = np.array(score_list).argmax()
        pred_label = label_list[target_id]
        if pred_label == 'NOT ENOUGH INFO':
            target_id = np.array(score_list).argsort()[1]
        pred_evi = evi_list[target_id]
        result.append({
            'id': idx,
            'label': label,
            'evidence': evidence_set,
            'predicted_label': pred_label,
            'predicted_evidence': pred_evi
        } if not is_test else {
            'id': idx,
            'predicted_label': pred_label,
            'predicted_evidence': pred_evi
        })
    return result


def precise(data,  is_test=False):
    result = []
    for idx, label, evidence_set, pred_result in data:
        label_list, evi_list, score_list = [], [], []
        for pred_label, score_seq, evi_seq in pred_result:
            label_list.append(pred_label)
            max_id = np.array(score_seq).argmax()
            score_list.append(score_seq[max_id])
            evi_list.append(evi_seq[max_id])
        target_id = np.array(score_list).argmax()
        pred_label = label_list[target_id]
        if pred_label == 'NOT ENOUGH INFO':
            target_id = np.array(score_list).argsort()[1]
        pred_evi = evi_list[target_id]
        result.append({
            'id': idx,
            'label': label,
            'evidence': evidence_set,
            'predicted_label': pred_label,
            'predicted_evidence': pred_evi
        } if not is_test else {
            'id': idx,
            'predicted_label': pred_label,
            'predicted_evidence': pred_evi
        })
    return result
