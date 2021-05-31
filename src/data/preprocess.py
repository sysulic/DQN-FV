#!/usr/bin/env python
# coding=utf-8
import json
import re
import sqlite3
from tqdm import tqdm
import unicodedata
from collections import defaultdict
import pdb

ENCODING = 'utf-8'
DATABASE = './data/fever/fever.db'
english_punctuations = {',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%'}

conn = sqlite3.connect(DATABASE)
cursor = conn.cursor()

def convert_string(string):
    string = re.sub('-LRB-', '(', string)
    string = re.sub('-RRB-', ')', string)
    string = re.sub('-LSB-', '[', string)
    string = re.sub('-RSB-', ']', string)
    string = re.sub('-LCB-', '{', string)
    string = re.sub('-RCB-', '}', string)
    string = re.sub('_', ' ', string)
    return string

def normalize(text: str) -> str:
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

def load_fake_evi_file(filename: str) -> dict:
    data = {}
    with open(filename, 'r') as fr:
        for line in fr:
            instance = json.loads(line.strip())
            if instance['label'] != 'NOT ENOUGH INFO': continue
            fake_evidence = sorted(instance['predicted_evidence'],
                                   key=lambda x: x[-1],
                                   reverse=True)
            data[instance['id']] = list(map(lambda x: [x[0], x[1]], fake_evidence))
    return data

def data_process(in_file: str, out_file: str,
                 fake_evi_file: str=None, is_fever2: bool=False,
                 fever2_helper_file: str=None) -> None:
    if in_file.find('train') != -1:
        mode = 'train'
        fake_evi = load_fake_evi_file(fake_evi_file)
    elif in_file.find('dev') != -1:
        mode = 'dev'
    else:
        mode = 'test'

    if is_fever2:
        fever2 = {}
        with open(fever2_helper_file, 'r') as fr:
            for line in fr:
                inst = json.loads(line.strip())
                fever2[inst['id']] = inst

    print(f'Loading {in_file}')
    instances = []
    with open(in_file, 'rb') as fr:
        for line in tqdm(fr.readlines()):
            instance = json.loads(line.decode(ENCODING).strip('\r\n'))
            
            if is_fever2:
                if 'original_id' not in instance:
                    continue
                else:
                    instance['predicted_pages'] = fever2[instance['original_id']]['predicted_pages']

            if mode == 'train':
                label = instance['label']
                evidence_set = []
                for evidence in instance['evidence']:
                    process_evidence = []
                    for sent in evidence:
                        if sent[2] is None: break
                        if [sent[2], sent[3]] in process_evidence: continue
                        process_evidence.append([sent[2], sent[3]])
                    if len(process_evidence) and process_evidence not in evidence_set:
                        evidence_set.append(process_evidence)
                if label == 'NOT ENOUGH INFO':
                    assert len(evidence_set) == 0
                    key = 'id' if not is_fever2 else 'original_id'
                    evidence_set.append(fake_evi[instance[key]])
            elif mode == 'dev':
                label = instance['label']
                evidence_set = instance['evidence']
            else:
                label = None
                evidence_set = None

            item = {
                'id': instance['id'],
                'label': label,
                'claim': instance['claim'],
                'evidence_set': evidence_set,
                'predicted_pages': instance['predicted_pages']
            }
            if is_fever2:
                item['original_id'] = instance['original_id']
                item['transformation'] = instance['transformation']
            instances.append(item)

    print(f'Processing and writing to {out_file}')
    with open(out_file, 'wb') as fw:
        for instance in tqdm(instances):
            titles = instance['predicted_pages']
            if mode == 'train':
                titles += [title for evidence in instance['evidence_set'] for title, _ in evidence]
            titles = list(set(titles))
            documents = defaultdict(dict)
            for title in titles:
                cursor.execute(
                    'SELECT * FROM documents WHERE id = ?',
                    (normalize(title),)
                )
                for row in cursor:
                    sentences = row[2].split('\n')
                    for sentence in sentences:
                        if sentence == '': continue
                        arr = sentence.split('\t')
                        if not arr[0].isdigit():
                            print(('Warning: this line from article %s for claim %d is not digit %s\r\n' % (title, instance['id'], sentence)).encode(ENCODING))
                            continue
                        line_num = int(arr[0])
                        if len(arr) <= 1: continue
                        sentence = ' '.join(arr[1:])
                        if sentence == '' or sentence in english_punctuations: continue
                        sentence = convert_string(normalize(sentence))
                        documents[title][line_num] = convert_string(normalize(title)) + ' ' + sentence
            documents = dict(documents)
            if len(documents) == 0: continue
            
            # 为 NEI 添加虚假证据
            if mode == 'train' and instance['label'] == 'NOT ENOUGH INFO':
                evidence_set = []
                assert len(instance['evidence_set']) == 1
                for title, num in instance['evidence_set'][0]:
                    if title in documents and int(num) in documents[title]:
                        evidence_set.append([[title, int(num)]])
                        break
                instance['evidence_set'] = evidence_set
            
            item = {
                'id': instance['id'],
                'claim': convert_string(normalize(instance['claim'])),
                'label': instance['label'],
                'evidence_set': instance['evidence_set'],
                'documents': documents
            }
            if is_fever2:
                item['original_id'] = instance['original_id']
                item['transformation'] = instance['transformation']

            fw.write((json.dumps(item) + '\n').encode(ENCODING))

if __name__ == '__main__':
    data_process('./data/retrieved/train.wiki7.jsonl', './data/dqn/train_v6.jsonl',
                 fake_evi_file='./data/retrieved/train.ensembles.s10.jsonl')
    data_process('./data/retrieved/dev.wiki7.jsonl', './data/dqn/dev_v6.jsonl')
    data_process('./data/retrieved/test.wiki7.jsonl', './data/dqn/test_v6.jsonl')
    #data_process('./data/fever/fever2-fixers-dev.jsonl',
    #             './data/dqn/fever2-dev_v6.jsonl',
    #             is_fever2=True,
    #             fever2_helper_file='./data/retrieved/dev.wiki7.jsonl')

