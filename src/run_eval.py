#!/usr/bin/env python
# coding=utf-8
import os
import argparse
import torch
import json
from torch.utils.data.dataloader import DataLoader
from run_train import evaluate, DQN_MODE
from post_processing.auto_thresholds_search import auto_threshold_search
from config import set_com_args, set_dqn_args, set_bert_args
from data.load_data import load_and_process_data
from eval.calc_score import calc_fever_score, calc_test_result, data_in_table
from logger import logger
from post_processing import predict_with_post_processing, predict_without_post_processing

def show_scores(scores, title):
    data = []
    for key in scores:
        data.append([key] + [round(value, 5)  for value in scores[key]])
    train_log = data_in_table(data, title=title) 
    logger.info('\n' + train_log)

def save_result(data, filename):
    with open(filename, 'w') as fw:
        for item in data:
            fw.write(json.dumps(item) + '\n')
    logger.info(f'Results are saved in {filename}')

def run_eval(args):
    Agent = DQN_MODE[args.dqn_mode]
    agent = Agent(args)
    agent.to(args.device)
    agent.load(args.checkpoint)
    eval_data = load_and_process_data(args,
                                      os.path.join(args.data_dir,
                                                   'dev_v6.jsonl' if args.do_eval else 'test_v6.jsonl'),
                                      agent.token)
    logger.info('[1] Searching thresholds')
    filepath = os.path.join(args.checkpoint, 'decision_seq_result.json')
    if not os.path.exists(filepath):
        logger.info('[1.1] Computing candidates of precise evidences and scores')
        dev_data = eval_data if args.do_eval else \
                load_and_process_data(args,
                                      os.path.join(args.data_dir, 'dev_v6.jsonl'),
                                      agent.token)
        evaluate(args, agent, args.checkpoint, dev_data, is_eval=True, print_log=False)
    alphas = auto_threshold_search(filepath)
    field_names, data = list(zip(*list(map(lambda x: ['alpha_%s' % x[0], x[1]],
                                           alphas['alpha'].items()))))
    title = 'LA: %.5f (T: %.5f, F: %.5f, N: %.5f)' % (alphas['LA']['total'], \
                                                      alphas['LA']['SUPPORTS'], \
                                                      alphas['LA']['REFUTES'], \
                                                      alphas['LA']['NOT ENOUGH INFO'])
    logger.info('Thresholds Results\n' + data_in_table([data], field_names=field_names, title=title))

    logger.info('[2] Computing target evidence and label')
    
    filepath = filepath if args.do_eval else os.path.join(args.checkpoint, 'test-decision_seq_result.json')
    logger.info('[2.1] Computing candidate precise evidences and scores')
    if not os.path.exists(filepath):
        evaluate(args, agent, args.checkpoint, eval_data, is_eval=args.do_eval, print_log=False)
    candidate_result = json.load(open(filepath, 'r'))

    logger.info('[2.2] Without post processing')
    _, precise_o = predict_without_post_processing(candidate_result, is_test=not args.do_eval)
    
    logger.info('[2.3] With post processing')
    precise_w = predict_with_post_processing(candidate_result, alphas['alpha'], is_test=not args.do_eval)

    if args.do_eval: # DEV
        logger.info('[2.4] Getting target evidence and label, and evaluation scores')
        result_precise_o, scores_o = calc_fever_score(precise_o, args.dev_true_file, logger=None)
        result_precise_w, scores_w = calc_fever_score(precise_w, args.dev_true_file, logger=None)
        save_result(result_precise_o, os.path.join(args.checkpoint, 'dev_precise_without_post_processing.jsonl'))
        save_result(result_precise_w, os.path.join(args.checkpoint, 'dev_precise_with_post_processing.jsonl'))
        show_scores(scores_o, 'Without Post-processing Strategy')
        show_scores(scores_w, 'With Post-processing Strategy')
    else:  # TEST
        logger.info('[2.4] Getting target evidence and label')
        result_precise_o = calc_test_result(precise_o, args.test_true_file, logger=None)
        result_precise_w = calc_test_result(precise_w, args.test_true_file, logger=None)
        save_result(result_precise_o, os.path.join(args.checkpoint, 'test_precise_without_post_processing.jsonl'))
        save_result(result_precise_w, os.path.join(args.checkpoint, 'test_precise_with_post_processing.jsonl'))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    set_com_args(parser)
    set_dqn_args(parser)
    set_bert_args(parser)
    args = parser.parse_args()
    args.logger = logger
    if args.model_type != 'roberta':
        args.label2id = {
            'NOT ENOUGH INFO': 2,
            'SUPPORTS': 1,
            'REFUTES': 0
        }
        args.id2label = ['REFUTES', 'SUPPORTS', 'NOT ENOUGH INFO']
    else:
        args.label2id = {
            'NOT ENOUGH INFO': 1,
            'SUPPORTS': 2,
            'REFUTES': 0
        }
        args.id2label = ['REFUTES', 'NOT ENOUGH INFO', 'SUPPORTS']
    args.do_lower_case = bool(args.do_lower_case)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.info(vars(args))
    
    run_eval(args)
