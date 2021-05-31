import json
#import pandas as pd
import itertools
import argparse
from tqdm import tqdm
import numpy as np

from logger import logger

def load_result(filename):
    with open(filename, 'r') as fr:
        data = json.load(fr)
    result = []
    for items in data:
        # idx, label, evidence_set
        idx, label = items[0], items[1]
        pred_result = {}
        for item in items[3]:
            pred_label = item[0]
            # [[q_sa, val_s],]
            score_seq = item[1]
            #evi_seq = [evi for _, _, _, evi in item]
            if pred_label == 'SUPPORTS':
                key = 'T'
            elif pred_label == 'REFUTES':
                key = 'F'
            else:
                key = 'N'
            pred_result[key] = score_seq
        result.append([idx, label, pred_result])
    return result


#def search_N(args):
#    points, samples, label_fn, filter_fn = args
#    best_value, target_alpha = float('-inf'), None
#    #for alpha in tqdm(points):
#    pre_score = sorted([(label_fn(*pred), label) for label, *pred in samples if filter_fn(*pred)])
#    index, N2N, TF2TF = 0, 0, 0
#    N_list, TF_list = [], []
#    for score, label in pre_score:
#        if label=="NOT ENOUGH INFO":
#            N2N += 1
#            N_list.append(score)
#        else:
#            TF_list.append(score)
#
#    for step, alpha in enumerate(points):
#        while index < len(pre_score) and pre_score[index][0]<=alpha:
#            if pre_score[index][1] == "NOT ENOUGH INFO":
#                N2N -= 1
#            if pre_score[index][1] != "NOT ENOUGH INFO":
#                TF2TF += 1
#            index += 1
#        if best_value <= N2N + TF2TF:
#            best_value = N2N + TF2TF
#            target_alpha = alpha
#        if step % 100 == 0:
#            logger.info(f'target_label:{"NOT ENOUGH INFO"}\tcur_best_value:{best_value}\ttarget_alpha:{target_alpha}')
#        if index >= len(pre_score):
#            break
#    logger.info(f'target_label:{"NOT ENOUGH INFO"}\tbest_value:{best_value}\ttarget_alpha:{target_alpha}')
#    return "NOT ENOUGH INFO", best_value, target_alpha
#
#def search_TF(args):
#    points, samples, target_label, label_fn, filter_fn = args
#    best_value, target_alpha = float('-inf'), None
#    #for alpha in tqdm(points):
#    pre_score=sorted([(label_fn(*pred), label) for label, *pred in samples if filter_fn(*pred)])
#    index, T2T, T2N=0, 0, 0
#    for score, label in pre_score:
#        if label==target_label:
#            T2T+=1
#        if label=="NOT ENOUGH INFO":
#            T2N+=1
#
#    for step, alpha in enumerate(points):
#        while index < len(pre_score) and pre_score[index][0]<=alpha:
#            if pre_score[index][1] == target_label:
#                T2T -= 1
#            if pre_score[index][1] == "NOT ENOUGH INFO":
#                T2N -= 1
#            index += 1
#        if best_value <= T2T - T2N:
#            best_value = T2T - T2N
#            target_alpha = alpha
#        if step % 100 == 0:
#            logger.info(f'target_label:{target_label}\tcur_best_value:{best_value}\ttarget_alpha:{target_alpha}')
#        if index >= len(pre_score):
#            break
#    logger.info(f'target_label:{target_label}\tbest_value:{best_value}\ttarget_alpha:{target_alpha}')
#    return target_label, best_value, target_alpha
#
#def calc_LA(samples, alpha_T, alpha_F, alpha_N):
#    prediction, labels = [], []
#    for label, (q_T, i), (q_F, j), (q_N, k) in samples:
#        value = min(q_N) - max(q_T[i], q_F[j])
#        if q_N[k] > max(q_T[i], q_F[j]) and value > alpha_N:
#            prediction.append('NOT ENOUGH INFO' == label)
#        elif q_T[i] > q_F[j]:
#            prediction.append('SUPPORTS' == label if q_T[i] - max(q_F[i], q_N[i]) > alpha_T else 'NOT ENOUGH INFO' == label)
#        else:
#            prediction.append('REFUTES' == label if q_F[j] - max(q_T[j], q_N[j]) > alpha_F else 'NOT ENOUGH INFO' == label)
#        labels.append(label)
#    LA = sum(prediction) / len(prediction)
#    LA_per_class = {'SUPPORTS': [0, 0], 'REFUTES': [0, 0], 'NOT ENOUGH INFO': [0, 0]}
#    for label, pred in zip(labels, prediction):
#        LA_per_class[label][0] += pred
#        LA_per_class[label][1] += 1
#    total_LA = {key: value[0] / value[1] for key, value in LA_per_class.items()}
#    total_LA['total'] = LA
#    return {'alpha': {'T': alpha_T, 'F': alpha_F, 'N': alpha_N}, 'LA': total_LA}
#
def search_N(args):
    points, samples, label_fn, filter_fn = args
    best_value, target_alpha = float('-inf'), None
    #for alpha in tqdm(points):
    pre_score=sorted([(label_fn(*pred), label) for label, *pred in samples if filter_fn(*pred)])
    index, N2N, TF2TF = 0, 0, 0
    N_list, TF_list = [], []
    for score, label in pre_score:
        if label == "NOT ENOUGH INFO":
            N2N += 1
            N_list.append(score)
        else:
        	TF_list.append(score)

    for step, alpha in enumerate(points):
        while index < len(pre_score) and pre_score[index][0] <= alpha:
            if pre_score[index][1] == "NOT ENOUGH INFO":
                N2N -= 1
            if pre_score[index][1]!="NOT ENOUGH INFO":
                TF2TF += 1
            index+=1
        if best_value <= N2N + TF2TF:
            best_value = N2N + TF2TF
            target_alpha = alpha
        if step % 100 == 0:
            logger.info(f'target_label:{"NOT ENOUGH INFO"}\tcur_best_value:{best_value}\ttarget_alpha:{target_alpha}')
        if index >= len(pre_score):
            break
    logger.info(f'target_label:{"NOT ENOUGH INFO"}\tbest_value:{best_value}\ttarget_alpha:{target_alpha}')
    return "NOT ENOUGH INFO", best_value, target_alpha

def search_TF(args):
    points, samples, target_label, label_fn, filter_fn = args
    best_value, target_alpha = float('-inf'), None
    #for alpha in tqdm(points):
    pre_score = sorted([(label_fn(*pred), label) for label, *pred in samples if filter_fn(*pred)])
    index, T2T, T2N = 0, 0, 0
    for score, label in pre_score:
        if label == target_label:
            T2T += 1
        if label == "NOT ENOUGH INFO":
            T2N += 1

    for step, alpha in enumerate(points):
        while index < len(pre_score) and pre_score[index][0] <= alpha:
            if pre_score[index][1] == target_label:
                T2T -= 1
            if pre_score[index][1] == "NOT ENOUGH INFO":
                T2N -= 1
            index += 1
        if best_value <= T2T - T2N:
            best_value = T2T - T2N
            target_alpha = alpha
        if step % 100 == 0:
            logger.info(f'target_label:{target_label}\tcur_best_value:{best_value}\ttarget_alpha:{target_alpha}')
        if index >= len(pre_score):
            break
    logger.info(f'target_label:{target_label}\tbest_value:{best_value}\ttarget_alpha:{target_alpha}')
    return target_label, best_value, target_alpha

def calc_LA(samples, alpha_T, alpha_F, alpha_N):
    prediction, labels = [], []
    for label, (q_T, i), (q_F, j), (q_N, k) in samples:
        #value=q_N[i] - q_T[i] if q_T[i]>q_F[j] else q_N[j]-q_F[j]
        value = min(q_N) - max(q_T[i], q_F[j])
        if q_N[k] > max(q_T[i], q_F[j]) and value > alpha_N:
            prediction.append('NOT ENOUGH INFO' == label)
        elif q_T[i] > q_F[j]:
            prediction.append('SUPPORTS' == label if q_T[i] - max(q_F[i], q_N[i]) > alpha_T else 'NOT ENOUGH INFO' == label)
        else:
            prediction.append('REFUTES' == label if q_F[j] - max(q_T[j], q_N[j]) > alpha_F else 'NOT ENOUGH INFO' == label)
        labels.append(label)
    LA = sum(prediction) / len(prediction)
    LA_per_class = {'SUPPORTS': [0, 0], 'REFUTES': [0, 0], 'NOT ENOUGH INFO': [0, 0]}
    for label, pred in zip(labels, prediction):
        LA_per_class[label][0] += pred
        LA_per_class[label][1] += 1
    total_LA = {key: value[0] / value[1] for key, value in LA_per_class.items()}
    total_LA['total'] = LA
    return {'alpha': {'T': alpha_T, 'F': alpha_F, 'N': alpha_N}, 'LA': total_LA}
#def auto_threshold_search(in_file, out_file):
def auto_threshold_search(in_file):
    raw_samples = load_result(in_file)
    samples = []

    for _, label, pred in raw_samples:
        pred = {key: np.asarray(value) for key, value in pred.items()}
        samples.append((
	    label,
	    (pred['T'], pred['T'].argmax()),
	    (pred['F'], pred['F'].argmax()),
	    (pred['N'], pred['N'].argmax()),
	))
    '''
    for _, label, pred in raw_samples:
	pred = {key: np.asarray(value) for key, value in pred.items()}
	samples.append((
     append       label,
	    ([pred['T'][-1]], 0),
	    ([pred['F'][-1]], 0),
	    ([pred['N'][-1]], 0),
	))
    ''' 

    N = np.asarray(sorted([min(q_N) - max(q_T[i], q_F[j]) \
                           for _, (q_T, i), (q_F, j), (q_N, k) in samples \
                               if q_N[k] > max(q_T[i], q_F[j])]))

    points_N = (N[:-1] + N[1:]) / 2

    alpha_N = search_N([points_N, samples,
                        lambda a, b, c: min(c[0]) - max(a[0][a[1]], b[0][b[1]]),
                        lambda a, b, c: c[0][c[1]] > max(a[0][a[1]], b[0][b[1]])])[-1]

    samples_TF = []

    for sample in samples:
            _, (q_T, i), (q_F, j), (q_N, k) = sample
            value = min(q_N) - max(q_T[i], q_F[j])
            if q_N[k] <= max(q_T[i], q_F[j]) or value <= alpha_N:
                    samples_TF.append(sample)

    T = np.asarray(sorted([q_T[i] - max(q_F[i], q_N[i]) for _, (q_T, i), (q_F, j), (q_N, k) in samples_TF if q_T[i] > q_F[j]]))
    F = np.asarray(sorted([q_F[j] - max(q_T[j], q_N[j]) for _, (q_T, i), (q_F, j), (q_N, k) in samples_TF if q_T[i] <= q_F[j]]))

    points_T = (T[:-1] + T[1:]) / 2
    points_F = (F[:-1] + F[1:]) / 2

    params = [[points_T, samples_TF, 'SUPPORTS',
               lambda a, b, c: a[0][a[1]] - max(b[0][a[1]], c[0][a[1]]),
               lambda a, b, _: a[0][a[1]] > b[0][b[1]]],
              [points_F, samples_TF, 'REFUTES',
               lambda a, b, c: b[0][b[1]] - max(a[0][b[1]], c[0][b[1]]),
               lambda a, b, _: a[0][a[1]] <= b[0][b[1]]]]

    alpha_T, alpha_F = search_TF(params[0])[-1], search_TF(params[1])[-1]

    LA_result = calc_LA(samples, alpha_T, alpha_F, alpha_N)

    #logger.info(f'Saving result to {args.out_file}')
    #with open(out_file, 'w') as fw:
    #    json.dump(LA_result, fw)
    #logger.info('LA result %s' % LA_result)
    return LA_result

