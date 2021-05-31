#!/usr/bin/env python
# coding=utf-8
import logging
import os
import argparse
import json
import pickle
import random
from typing import List, Tuple
from tqdm import tqdm, trange
from time import sleep
from functools import reduce
#from itertools import chain
from collections import defaultdict
from copy import deepcopy
import pdb
#from multiprocessing import cpu_count, Pool

import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np

from data.load_data import load_and_process_data
from dqn.lstm_dqn import LstmDQN
from dqn.transformer_dqn import TransformerDQN
from environment import BaseEnv, FeverEnv
from replay_memory import ReplayMemory, PrioritizedReplayMemory, ReplayMemoryWithLabel, PrioritizedReplayMemoryWithLabel
from data.structure import *
from data.dataset import collate_fn, FeverDataset
from config import set_com_args, set_dqn_args, set_bert_args
from eval.calc_score import calc_fever_score, calc_test_result, data_in_table
from post_processing import predict_with_post_processing, predict_without_post_processing

#logger = logging.getLogger(__name__)
from logger import logger

#Agent = BertDQN
DQN_MODE = {
    'lstm': LstmDQN,
    'transformer': TransformerDQN
}
Memory = {
    'random': ReplayMemory,
    'priority': PrioritizedReplayMemory,
    'label_random': ReplayMemoryWithLabel,
    'label_priority': PrioritizedReplayMemoryWithLabel
}


def set_random_seeds(random_seed):
    """Sets all possible random seeds so results can be reproduced"""
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_seed)
    # tf.set_random_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)


def train(args,
          agent,
          train_dataset: FeverDataset,
          epochs_trained: int=0,
          acc_loss_trained_in_current_epoch: float=0,
          steps_trained_in_current_epoch: int=0,
          losses_trained_in_current_epoch: List[float]=[]) -> None:

    def save_train_log(scores, loss, epoch, step):
        title = 'epoch=%d,step=%d,loss=%.5f' % (epoch, step, loss)
        data = []
        for key in scores:
            data.append([key] + [round(value, 5)  for value in scores[key]])
        train_log = data_in_table(data, title=title) 
        with open(os.path.join(args.output_dir, 'train_log.txt'), 'a') as fw:
            fw.write(train_log + '\n')
        logger.info('\n' + train_log)
        
    logger.info('Training')
    env = FeverEnv(label2id=args.label2id, K=args.max_evi_size)
    if args.mem.find('label') == -1:
        memory = Memory[args.mem](args.capacity)
    else:
        memory = Memory[args.mem](args.capacity, args.num_labels, args.proportion)
    
    data_loader = DataLoader(train_dataset,
                             #num_workers=1,
                             num_workers=0,
                             collate_fn=collate_fn,
                             batch_size=args.train_batch_size,
                             shuffle=True)
    #train_iterator = trange(int(args.num_train_epochs), desc='Epoch', disable=args.local_rank not in [-1, 0])
    train_iterator = trange(int(args.num_train_epochs), desc='Epoch')
    for epoch in train_iterator:
        if epochs_trained > 0:
            epochs_trained -= 1
            sleep(0.1)
            continue
        
        epoch_iterator = tqdm(data_loader,
                              desc='Loss')
        
        #log_per_steps = len(epoch_iterator) // 5
        log_per_steps = 100

        t_loss, t_steps = acc_loss_trained_in_current_epoch, steps_trained_in_current_epoch
        t_losses, losses = losses_trained_in_current_epoch, []

        for step, (batch_state, batch_actions) in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            cur_iter, tmp_iter = 0, random.randint(1, 5)
            while True:
                batch_selected_action, _  = agent.select_action(batch_state,
                                                                  batch_actions,
                                                                  net=agent.q_net,
                                                                  is_greedy=False)
                batch_state_next, batch_actions_next = [], []
                for state, selected_action, actions in zip(batch_state,
                                                           batch_selected_action,
                                                           batch_actions):
                    state_next, reward, done = env.step(state, selected_action)
                    actions_next = \
                            list(filter(lambda x: selected_action.sentence.id != x.sentence.id, actions))
                    done = done if len(actions_next) else True
                    
                    data = {'item': Transition(state=state,
                                               action=selected_action,
                                               next_state=state_next,
                                               reward=reward,
                                               next_actions=actions_next,
                                               done=done)}
                    if args.mem.find('label') != -1:
                        data['label'] = state.label
                    memory.push(**data)
                    
                    if done: continue
                    
                    batch_state_next.append(state_next)
                    batch_actions_next.append(actions_next)

                batch_state = batch_state_next
                batch_actions = batch_actions_next
                cur_iter += 1
                # sample batch data and optimize model
                if len(memory) >= args.train_batch_size:
                    if args.mem.find('priority') != -1:
                        #tree_idx, isweights, batch_rl = memory.sample(args.train_batch_size)
                        tree_idx, batch_rl = memory.sample(args.train_batch_size)
                    else:
                        batch_rl = memory.sample(args.train_batch_size)
                        #isweights = None
                    isweights = None
                    td_error, [loss, wloss] = agent.update(batch_rl,
                                                           isweights,
                                                           log=step and step % log_per_steps == 0 and cur_iter % tmp_iter == 0)
                    if args.mem.find('priority') != -1:
                        #errors = [e for i, e in enumerate(loss.tolist()) if flag[i]]
                        memory.batch_update_sumtree(tree_idx, td_error)
                    t_loss += loss
                    t_steps += 1
                    losses.append(loss)
                    epoch_iterator.set_description('%.4f (%.4f)' % (wloss, loss))
                    epoch_iterator.refresh()
                
                if len(batch_state) == 0: break

            if step and step % args.target_update == 0:
                agent.soft_update_of_target_network(args.tau)
            
            if step and step % args.save_steps == 0:
                save_dir = os.path.join(args.output_dir, f'{epoch}-{step}-{t_loss / t_steps}')
                agent.save(save_dir)
                with open(os.path.join(save_dir, 'memory.pk'), 'wb') as fw:
                    pickle.dump(memory, fw)
                with open(os.path.join(save_dir, 'loss.txt'), 'w') as fw:
                    fw.write('\n'.join(list(map(str, losses))))
                t_losses.extend(losses)
                losses = []
                
                if args.do_eval:
                    scores = evaluate(args, agent, save_dir, print_log=True)
                    save_train_log(scores, t_loss / t_steps, epoch, step)

        epoch_iterator.close()

        acc_loss_trained_in_current_epoch = 0
        losses_trained_in_current_epoch = []
        
        save_dir = os.path.join(args.output_dir, f'{epoch + 1}-0-{t_loss / t_steps}')
        if steps_trained_in_current_epoch == 0:
            agent.save(save_dir)
            with open(os.path.join(save_dir, 'memory.pk'), 'wb') as fw:
                pickle.dump(memory, fw)
            with open(os.path.join(save_dir, 'loss.txt'), 'w') as fw:
                fw.write('\n'.join(list(map(str, losses))))
            losses = []
        
        scores = evaluate(args, agent, save_dir, print_log=True)
        save_train_log(scores, t_loss / t_steps, epoch + 1, 0)
                
    train_iterator.close()


def evaluate(args: dict, agent, save_dir: str, dev_data: FeverDataset=None, is_eval: bool=True, print_log: bool=False):
    agent.eval()
    if dev_data is None:
        if args.do_eval:
            filename = 'dev_v6.jsonl'
        elif args.do_test:
            filename = 'test_v6.jsonl'
        dev_data = load_and_process_data(args,
                                         os.path.join(args.data_dir, filename),
                                         agent.token)
    data_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=1, shuffle=False)
    epoch_iterator = tqdm(data_loader,
                          disable=args.local_rank not in [-1, 0])
    results_of_q_state_seq = []
    results = []
    logger.info('Evaluating')
    with torch.no_grad():
        for batch_state, batch_actions in epoch_iterator:
            q_value_seq, state_seq = [], []
            for _ in range(args.max_evi_size):
                batch_selected_action, batch_q_value = \
                        agent.select_action(batch_state,
                                            batch_actions,
                                            net=agent.q_net,
                                            is_greedy=True)
                
                batch_state_next, batch_actions_next = [], []
                for state, selected_action, actions in zip(batch_state,
                                                           batch_selected_action,
                                                           batch_actions):
                    state_next = BaseEnv.new_state(state, selected_action)
                    actions_next = \
                            list(filter(lambda x: selected_action.sentence.id != x.sentence.id, actions))
                    
                    batch_state_next.append(state_next)
                    
                    if len(actions_next) == 0:
                        continue
                    else:
                        batch_actions_next.append(actions_next)
                
                q_value_seq.append(batch_q_value)
                state_seq.append(batch_state_next)
                
                if len(batch_actions_next) == 0:
                    break

                batch_state = batch_state_next
                batch_actions = batch_actions_next
            
            idx = state_seq[0][0].claim.id
            label = args.id2label[state_seq[0][0].label] if is_eval else None
            evidence_set = state_seq[0][0].evidence_set if is_eval else None
            pred_labels = [args.id2label[state.pred_label] for state in state_seq[0]]
            score_seq_of_three_labels = list(zip(*q_value_seq))
            evi_seq_of_three_labels = list(zip(*[[reduce(lambda seq1, seq2: seq1 + seq2,
                                                         map(lambda sent: [list(sent.id)],
                                                             state.candidate)) if state.candidate else [] \
                                                  for state in batch_state] for batch_state in state_seq]))
            pred_result = list(zip(pred_labels, score_seq_of_three_labels, evi_seq_of_three_labels))

            results_of_q_state_seq.append([idx, label, evidence_set, pred_result])

    name = 'decision_seq_result.json'
    if not is_eval:
        name = f'test-{name}'
    with open(os.path.join(save_dir, name), 'w') as fw:
        json.dump(results_of_q_state_seq, fw)

    if args.do_eval and print_log:
        results, _ = predict_without_post_processing(results_of_q_state_seq)
        _, scores = calc_fever_score(results, args.dev_true_file, logger=None)
        
        return scores


def run_dqn(args) -> None:
    Agent = DQN_MODE[args.dqn_mode]
    agent = Agent(args)
    agent.to(args.device)
    
    train_dataset = load_and_process_data(args,
                                       os.path.join(args.data_dir, 'train_v6.jsonl'),
                                       agent.token)
    epochs_trained = 0
    acc_loss_trained_in_current_epoch = 0
    steps_trained_in_current_epoch = 0
    losses_trained_in_current_epoch = []
    if args.checkpoint:
        names = list(filter(lambda x: x != '', args.checkpoint.split('/')))[-1].split('-')
        epochs_trained = int(names[0])
        steps_trained_in_current_epoch = int(names[1])
        acc_loss_trained_in_current_epoch = float('.'.join(names[2].split('.')[:-1])) * steps_trained_in_current_epoch
        agent.load(args.checkpoint)
        with open(os.path.join(args.checkpoint, 'memory.pk'), 'rb') as fr:
            memory = pickle.load(fr)
        with open(os.path.join(args.checkpoint, 'loss.txt'), 'r') as fr:
            losses_trained_in_current_epoch = list(map(float, fr.readlines()))
    train(args,
          agent,
          train_dataset,
          epochs_trained,
          acc_loss_trained_in_current_epoch,
          steps_trained_in_current_epoch,
          losses_trained_in_current_epoch)


def main() -> None:
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

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

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

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    #args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    logger.info(vars(args))
    
    # Set seed
    set_random_seeds(args.seed)

    #if args.local_rank not in [-1, 0]:
    #    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    logger.info("Training/evaluation parameters %s", args)

    # run dqn
    run_dqn(args)

if __name__ == '__main__':
    main()

