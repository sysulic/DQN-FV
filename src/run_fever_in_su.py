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
from itertools import chain
from collections import defaultdict
from copy import deepcopy
import pdb
#from multiprocessing import cpu_count, Pool

import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
import numpy as np

#from dqn.lstm_dqn import QNetwork
from dqn.transformer_dqn import QNetwork
from data.structure import *
from data.dataset_in_su import collate_fn_for_train, collate_fn_for_predict, FeverDataset
from eval.calc_score import calc_fever_score, truncate_q_values, calc_test_result
from config import set_com_args, set_dqn_args, set_bert_args

logger = logging.getLogger(__name__)

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
          train_dataset: FeverDataset,
          epochs_trained: int=0,
          acc_loss_trained_in_current_epoch: float=0,
          steps_trained_in_current_epoch: int=0,
          losses_trained_in_current_epoch: List[float]=[]) -> None:

    def log_scores(scores, loss, epoch, step):
        content = '*' * 10 + 'epoch=%d,step=%d,loss=%.5f' % (epoch, step, loss) + '*' * 10 + '\n'
        for key in scores:
            content += 'key=%s\n' % key
            for label in scores[key]:
                #strict_score, label_accuracy, precision, recall, f1 = scores[thred][label]
                content += f'{label}\t{scores[key][label]}\n'
        content += '\n'
        with open(os.path.join(args.output_dir, 'results.txt'), 'a') as fw:
            fw.write(content)
        
    logger.info('Training')

    model = QNetwork(num_labels=args.num_labels,
                     hidden_size=1024,
                     nheads=8,
                     num_layers=args.num_layers)
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    if args.checkpoint:
        state_dict = torch.load(os.path.join(args.checkpoint, 'model.bin'),
                                map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
    
    data_loader = DataLoader(train_dataset,
                             num_workers=1,
                             #num_workers=0,
                             collate_fn=collate_fn_for_train,
                             batch_size=args.train_batch_size,
                             shuffle=True)
    train_iterator = trange(int(args.num_train_epochs), desc='Epoch')
    for epoch in train_iterator:
        if epochs_trained > 0:
            epochs_trained -= 1
            sleep(0.1)
            continue
        
        epoch_iterator = tqdm(data_loader, desc='Loss')
        
        log_per_steps = len(epoch_iterator) // 5

        t_loss, t_steps = acc_loss_trained_in_current_epoch, steps_trained_in_current_epoch
        t_losses, losses = losses_trained_in_current_epoch, []

        for step, (b_clm, b_evi, b_mask, b_jaccard) in enumerate(epoch_iterator):
            if step == 3216: pdb.set_trace()
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            inputs = {
                'claims': b_clm.to(args.device),
                'evidences': b_evi.to(args.device),
                'evidences_mask': b_mask.to(args.device)
            }
            outputs = torch.tanh(model(**inputs)[0])
            b_jaccard = b_jaccard.to(args.device)
            loss = F.mse_loss(outputs, b_jaccard)
            loss.backward()
            optimizer.step()
            model.zero_grad()

            if step % log_per_steps == 0:
                ids = random.sample(range(outputs.size(0)), k=min(5, outputs.size(0)))
                y_hat, y = outputs[ids], b_jaccard[ids]
                print(torch.cat([y.unsqueeze(2), y_hat.unsqueeze(2)], dim=-1))
            
            losses.append(loss.detach().cpu().item())
            epoch_iterator.set_description('%.4f' % losses[-1])
            epoch_iterator.refresh()

        epoch_iterator.close()

        acc_loss_trained_in_current_epoch = 0
        losses_trained_in_current_epoch = []
        
        save_dir = os.path.join(args.output_dir, f'{epoch + 1}-0-{np.array(losses).mean()}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        if steps_trained_in_current_epoch == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(save_dir, 'model.bin'))
            with open(os.path.join(save_dir, 'loss.txt'), 'w') as fw:
                fw.write('\n'.join(list(map(str, losses))))
        
        if args.do_eval:
            scores = evaluate(args, model, save_dir)
            log_scores(scores, np.array(losses).mean(), epoch + 1, 0)
        
        losses = []
    
    train_iterator.close()


def evaluate(args: dict, model, save_dir: str, dev_data: FeverDataset=None, is_eval: bool=True):
    def convert_input(clm, evi, sents):
        batch, seq, dim = len(sents), len(evi) + 1, len(clm)

        clm_tensor = torch.stack([clm] * batch)
        evi_tensor = torch.stack([torch.stack(evi + [sent]) for sent in sents])
        evi_mask = torch.ones(batch, seq, dtype=torch.float)

        assert clm_tensor.size() == torch.Size((batch, dim))
        assert evi_tensor.size() == torch.Size((batch, seq, dim))
        assert evi_mask.size() == torch.Size((batch, seq))
        
        return clm_tensor, evi_tensor, evi_mask

    model.eval()
    if dev_data is None:
        if args.do_eval:
            filename = 'cached_dev_roberta-large_v5+6'
        elif args.do_test:
            filename = 'cached_test_roberta-large_v5+6'
        dev_data = FeverDataset(os.path.join(args.data_dir, filename),
                                args.label2id,
                                is_train=False)
    data_loader = DataLoader(dev_data, collate_fn=collate_fn_for_predict, batch_size=1, shuffle=False)
    epoch_iterator = tqdm(data_loader)
    results_seq = []
    logger.info('Evaluating')
    with torch.no_grad():
        for clm_id, sent_ids, clm_tensor, sent_tensor, label, evidence_set in epoch_iterator:
            jaccard_seq, label_seq, evi_seq = [], [], []
            pred_evi = []
            for _ in range(args.max_evi_size):
                clm, evi, mask = convert_input(clm_tensor, pred_evi, sent_tensor)
                inputs = {
                    'claims': clm.to(args.device),
                    'evidences': evi.to(args.device),
                    'evidences_mask': mask.to(args.device)
                }
                outputs = torch.tanh(model(**inputs)[0])

                ids, v = outputs.argmax(), outputs.max()
                i, j = ids // len(args.label2id), ids % len(args.label2id)
                p_jaccard, p_sent_id, p_label = outputs[i][j], sent_ids[i], j
                assert p_jaccard == outputs[i][j]
                
                jaccard_seq.append(p_jaccard.item())
                label_seq.append(args.id2label[p_label])
                evi_seq.append(list(p_sent_id))
                
                pred_evi.append(sent_tensor[i])
                sent_ids = [ids for k, ids in enumerate(sent_ids) if k != i]
                sent_tensor = [sent for k, sent in enumerate(sent_tensor) if k != i]

                if len(sent_ids) == 0: break

            results_seq.append([clm_id, args.id2label[label], evidence_set, \
                               jaccard_seq, label_seq, evi_seq])

    name = 'decision_seq_result.json'
    if not is_eval:
        name = f'test-{name}'
    with open(os.path.join(save_dir, name), 'w') as fw:
        json.dump(results_seq, fw)

    def get_result(is_test, is_precise):
        results = []
        for clm_id, label, evidence_set, jaccard_seq, label_seq, evi_seq in results_seq:
            if is_precise:
                t = np.asarray(jaccard_seq).argmax()
                pred_label, pred_evi = label_seq[t], evi_seq[:t + 1]
            else:
                pred_label, pred_evi = label_seq[-1], evi_seq
            results.append({
                'id': clm_id,
                'label': label,
                'evidence': evidence_set,
                'predicted_label': pred_label,
                'predicted_evidence': pred_evi
            } if not is_test else {
                'id': clm_id,
                'predicted_label': pred_label,
                'predicted_evidence': pred_evi
            })
        return results

    if not is_eval:
        results = get_result(is_test=True, is_precise=False)
        predicted_list = calc_test_result(results, args.test_true_file, logger=None)
        with open(os.path.join(save_dir, 'test-imprecise.jsonl'), 'w') as fw:
            for item in predicted_list:
                fw.write(json.dumps(item) + '\n')
        
        results = get_result(is_test=True, is_precise=True)
        predicted_list = calc_test_result(results, args.test_true_file, logger=None)
        with open(os.path.join(save_dir, 'test-precise.jsonl'), 'w') as fw:
            for item in predicted_list:
                fw.write(json.dumps(item) + '\n')
        logger.info(f'Testing result is saved in {save_dir}')
        return
    
    if args.do_eval:
        thred_results = defaultdict(dict)
        results = get_result(is_test=False, is_precise=False)
        predicted_list, scores = calc_fever_score(results, args.dev_true_file, logger=None)
        thred_results['scores']['imprecise'] = scores
        thred_results['predicted_list']['imprecise'] = predicted_list
        
        results = get_result(is_test=False, is_precise=True)
        predicted_list, scores = calc_fever_score(results, args.dev_true_file, logger=None)
        thred_results['scores']['precise'] = scores
        thred_results['predicted_list']['precise'] = predicted_list
        
        thred_results = dict(thred_results)
        
        filename = 'eval.json'
        with open(os.path.join(save_dir, filename), 'w') as fw:
            json.dump(thred_results, fw)
        logger.info(f'Results are saved in {os.path.join(save_dir, filename)}')
        
        return thred_results['scores']


def run_dqn(args) -> None:
    if args.do_train:
        train_dataset = FeverDataset(os.path.join(args.data_dir, 'cached_train_roberta-large_v5+6'),
                                     args.label2id)
        epochs_trained = 0
        acc_loss_trained_in_current_epoch = 0
        steps_trained_in_current_epoch = 0
        losses_trained_in_current_epoch = []
        if args.checkpoint:
            names = list(filter(lambda x: x != '', args.checkpoint.split('/')))[-1].split('-')
            epochs_trained = int(names[0])
            steps_trained_in_current_epoch = int(names[1])
            acc_loss_trained_in_current_epoch = float('.'.join(names[2].split('.')[:-1])) * steps_trained_in_current_epoch
            with open(os.path.join(args.checkpoint, 'loss.txt'), 'r') as fr:
                losses_trained_in_current_epoch = list(map(float, fr.readlines()))
        train(args,
              train_dataset,
              epochs_trained,
              acc_loss_trained_in_current_epoch,
              steps_trained_in_current_epoch,
              losses_trained_in_current_epoch)
        
    elif args.do_eval or args.do_test or args.do_fever2:
        assert args.checkpoints is not None
        model = QNetwork(num_labels=args.num_labels,
                         hidden_size=1024,
                         nheads=8,
                         num_layers=args.num_layers)
        for checkpoint in args.checkpoints:
            state_dict = torch.load(os.path.join(checkpoint, 'model.bin'),
                                    map_location=torch.device('cpu'))
            model.load_state_dict(state_dict['model'])
            if args.do_eval:
                evaluate(args, model, checkpoint, dev_data, is_eval=True)
            if args.do_test:
                evaluate(args, model, checkpoint, test_data, is_eval=False)


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
    logger.info(vars(args))

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
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        #"Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        #args.fp16,
    )

    # Set seed
    set_random_seeds(args.seed)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    logger.info("Training/evaluation parameters %s", args)

    # run dqn
    run_dqn(args)

if __name__ == '__main__':
    main()

