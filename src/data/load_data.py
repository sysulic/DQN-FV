#!/usr/bin/env python3
# coding=utf-8
from tqdm import tqdm, trange
from typing import List, Tuple
import os
import json
import pickle

import torch

from data.structure import *
from data.dataset import FeverDataset

from transformers import (
    AlbertConfig,
    AlbertModel,
    AlbertTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    XLNetConfig,
    XLNetTokenizer,
    #XLNetModel,
    XLNetForSequenceClassification,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            AlbertConfig,
            RobertaConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    #"xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

def initilize_bert(args):
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    config.num_labels = args.num_labels
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    model.to(args.device)

    def model_output(**params):
        if args.model_type in {'bert', 'albert'}:
            return model(**params)[1]
        elif args.model_type in {'xlnet'}:
            output = model.transformer(**params)[0]
            return model.sequence_summary(output)
        elif args.model_type in {'roberta'}:
            output = model.roberta(**params)[0]
            output = output[:, 0, :]  # take <s> token (equiv. to [CLS])
            return output
    
    def feature_extractor(texts: List[str]) -> List[List[float]]:
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        pad_token_segment_id = 4 if args.model_type in ['xlnet'] else 0
        pad_on_left = bool(args.model_type in ['xlnet'])
        
        texts = [texts[0]] + [[texts[0], text] for text in texts[1:]]
        inputs = tokenizer.batch_encode_plus(texts, max_length=256)
        # padding
        max_length = max([len(input_ids) for input_ids in inputs['input_ids']])
        if pad_on_left:
            inputs['input_ids'] = torch.tensor(
                [[pad_token] * (max_length - len(input_ids)) + input_ids for input_ids in inputs['input_ids']],
                dtype=torch.long
            )
            inputs['attention_mask'] = torch.tensor(
                [[0] * (max_length - len(mask)) + mask for mask in inputs['attention_mask']],
                dtype=torch.long
            )
            inputs['token_type_ids'] = torch.tensor(
                [[pad_token_segment_id] * (max_length - len(token_type)) + token_type \
                 for token_type in inputs['token_type_ids']],
                dtype=torch.long
            ) if args.model_type in ['bert', 'xlnet', 'albert'] else None
        else:
            inputs['input_ids'] = torch.tensor(
                [input_ids + [pad_token] * (max_length - len(input_ids)) for input_ids in inputs['input_ids']],
                dtype=torch.long
            )
            inputs['attention_mask'] = torch.tensor(
                [mask + [0] * (max_length - len(mask)) for mask in inputs['attention_mask']],
                dtype=torch.long
            )
            inputs['token_type_ids'] = torch.tensor(
                [token_type + [pad_token_segment_id] * (max_length - len(token_type)) \
                 for token_type in inputs['token_type_ids']],
                dtype=torch.long
            ) if args.model_type in ['bert', 'xlnet', 'albert'] else None
        
        with torch.no_grad():
            INTERVEL = 64
            outputs = [model_output(
                            **dict(map(lambda x: (x[0], x[1][i:i + INTERVEL].to(args.device) if x[1] is not None else x[1]),
                                       inputs.items()))
                        ) for i in range(0, inputs['input_ids'].size(0), INTERVEL)]
            outputs = torch.cat(outputs, dim=0)
            assert outputs.size(0) == inputs['input_ids'].size(0)
        return outputs.detach().cpu().numpy().tolist()
    
    return feature_extractor

def load_and_process_data(args: dict, filename: str, token_fn: 'function', fake_evi: bool=False, is_raw: bool=False) \
        -> DataSet:
    if filename.find('train') != -1:
        mode = 'train'
    elif filename.find('dev') != -1:
        mode = 'dev'
    else:
        mode = 'test'
    cached_file = os.path.join(
        '/'.join(filename.split('/')[:-1]),
        'cached_{}_{}_v5+6'.format(
            mode,
            list(filter(None, args.model_name_or_path.split('/'))).pop()
        )
    )
    
    data = None
    if not os.path.exists(cached_file):
        feature_extractor = initilize_bert(args)

        os.makedirs(cached_file, exist_ok=True)
        
        args.logger.info(f'Loading and processing data from {filename}')
        data = []
        skip, count, num = 0, 0, 0
        with open(filename, 'rb') as fr:
            for line in tqdm(fr.readlines()):
                instance = json.loads(line.decode('utf-8').strip())
                
                total_texts = [sentence for _, text in instance['documents'].items() \
                                            for _, sentence in text.items()]
                if mode == 'train' and len(total_texts) < 5:
                    skip += 1
                    continue
                count += 1
                
                total_texts = [instance['claim']] + total_texts
                semantic_embedding = feature_extractor(total_texts)
                
                claim = Claim(id=instance['id'],
                              str=instance['claim'],
                              tokens=semantic_embedding[0])
                sent2id = {}
                sentences = []
                text_id = 1
                for title, text in instance['documents'].items():
                    for line_num, sentence in text.items():
                        sentences.append(Sentence(id=(title, int(line_num)),
                                                  str=sentence,
                                                  tokens=semantic_embedding[text_id]))
                        sent2id[(title, int(line_num))] = len(sentences) - 1
                        text_id += 1
                assert text_id == len(semantic_embedding)
                
                if mode == 'train':
                    label = args.label2id[instance['label']]
                    evidence_set = [[sentences[sent2id[(title, int(line_num))]] \
                                        for title, line_num in evi] \
                                            for evi in instance['evidence_set']]
                    if not fake_evi and instance['label'] == 'NOT ENOUGH INFO':
                        evidence_set = []
                elif mode.find('dev') != -1:
                    label = args.label2id[instance['label']]
                    evidence_set = instance['evidence_set']
                else:
                    label = evidence_set = None
                data.append((claim, label, evidence_set, sentences))
                
                if count % 1000 == 0:
                    for item in data:
                        with open(os.path.join(cached_file, f'{num}.pk'), 'wb') as fw:
                            pickle.dump(item, fw)
                        num += 1
                    data = []

            for item in data:
                with open(os.path.join(cached_file, f'{num}.pk'), 'wb') as fw:
                    pickle.dump(item, fw)
                num += 1
            del data
        args.logger.info(f'Process Done. Skip: {skip}({skip / count})')

    dataset = FeverDataset(cached_file, label2id=args.label2id)
    return dataset
