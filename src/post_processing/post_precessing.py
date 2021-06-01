#!/usr/bin/env python
# coding=utf-8
import numpy as np

def imprecise(data, alphas, is_test=False):
    alpha_T, alpha_F, alpha_N = alphas['T'], alphas['F'], alphas['N']
    result = []
    for idx, label, evidence_set, pred_result in data:
        label_map = {v: i for i, v in enumerate(pred_label for pred_label, *_ in pred_result)}
        T, F, N = label_map['SUPPORTS'], label_map['REFUTES'], label_map['NOT ENOUGH INFO']
        q_T, q_F, q_N = [[pred_result[i][1][-1]] for i in [T, F, N]]
        i, j, k = [np.array(x).argmax() for x in [q_T, q_F, q_N]]
        if q_N[k] > max(q_T[i], q_F[j]) and min(q_N) - max(q_T[i], q_F[j]) > alpha_N:
            pred_label = 'NOT ENOUGH INFO'
            pred_evi=pred_result[T][-1][-1] if q_T[i]>q_F[j] else pred_result[F][-1][-1]
        elif q_T[i] > q_F[j]:
            pred_evi = pred_result[T][-1][-1]
            if q_T[i] - max(q_F[i], q_N[i]) > alpha_T:
                pred_label = 'SUPPORTS'
            else:
                pred_label = 'NOT ENOUGH INFO'
        else:
            pred_evi = pred_result[F][-1][-1]
            if q_F[j] - max(q_T[j], q_N[j]) > alpha_F:
                pred_label = 'REFUTES'
            else:
                pred_label = 'NOT ENOUGH INFO'
                
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


def precise(data,  alphas,is_test=False):
    alpha_T, alpha_F, alpha_N = alphas['T'], alphas['F'], alphas['N']
    result = []
    for idx, label, evidence_set, pred_result in data:
        label_map = {v: i for i, v in enumerate(pred_label for pred_label, *_ in pred_result)}
        T, F, N = label_map['SUPPORTS'], label_map['REFUTES'], label_map['NOT ENOUGH INFO']
        q_T, q_F, q_N = [pred_result[i][1] for i in [T, F, N]]
        i, j, k = [np.array(x).argmax() for x in [q_T, q_F, q_N]]
        if q_N[k] > max(q_T[i], q_F[j]) and min(q_N) - max(q_T[i], q_F[j]) > alpha_N:
            pred_label = 'NOT ENOUGH INFO'
            pred_evi = pred_result[T][-1][i] if q_T[i]>q_F[j] else pred_result[F][-1][j]
        elif q_T[i] > q_F[j]:
            pred_evi = pred_result[T][-1][i]
            if q_T[i] - max(q_F[i], q_N[i]) > alpha_T:
                pred_label = 'SUPPORTS'
            else:
                pred_label = 'NOT ENOUGH INFO'
        else:
            pred_evi = pred_result[F][-1][j]
            if q_F[j] - max(q_T[j], q_N[j]) > alpha_F:
                pred_label = 'REFUTES'
            else:
                pred_label = 'NOT ENOUGH INFO'
                
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
