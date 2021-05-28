# coding: utf-8
#import csv
import argparse
from pprint import pprint
from tqdm import tqdm
import sqlite3
import unicodedata
from collections import defaultdict
import json
import re
import pdb

ENCODING = 'utf-8'
DATABASE = './data/fever/fever.db'
english_punctuations = {',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%'}

conn = sqlite3.connect(DATABASE)
cursor = conn.cursor()
#import sys

def convert_string(string):
    string = re.sub('-LRB-', '(', string)
    string = re.sub('-RRB-', ')', string)
    string = re.sub('-LSB-', '[', string)
    string = re.sub('-RSB-', ']', string)
    string = re.sub('-LCB-', '{', string)
    string = re.sub('-RCB-', '}', string)
    string = re.sub('_', ' ', string)
    return string

#csv.field_size_limit(sys.maxsize)
def normalize(text: str) -> str:
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

def process_data(instances):
    new_instances = []
    for instance in tqdm(instances):
        if instance['label'] != 'NOT ENOUGH INFO':
            evidence_set = set(map(lambda evidence: tuple(map(lambda sent: (sent[2], sent[3]),
                                                              evidence)),
                                   instance['evidence']))
        else:
            evidence_set = set(map(lambda sent: ((sent[0], sent[1]),), instance['predicted_evidence']))
        titles = set([normalize(sent[0]) for evidence in evidence_set for sent in evidence])
        
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
                    documents[title][line_num] = convert_string(title) + ' ' + sentence
        documents = dict(documents)

        for evidence in evidence_set:
            if len(evidence) > 1: continue
            #evi_str = ' '.join([documents[title][line_num] for title, line_num in evidence[:5] \
            #                   if title in documents and line_num in documents[title]])
            evi_str = ' '.join([documents[title][line_num] for title, line_num in evidence \
                               if title in documents and line_num in documents[title]])
            if evi_str == '': continue
            #label = instance['label'] if len(evidence) <= 5 else 'NOT ENOUGH INFO'
            label = instance['label']
            new_instances.append({
                'id': str(instance['id']),
                'claim': convert_string(normalize(instance['claim'])),
                'evidence': evi_str,
                'label': label
            })
    return new_instances


def read_jsonl(filename):
    print(f'reading {filename}')
    instances = []
    with open(filename, 'r') as fr:
        for line in tqdm(fr.readlines()):
            instances.append(json.loads(line.strip()))
    return instances

def write_csv(filename, data):
    print(f'writting to {filename}')
    outputs = []
    for items in data:
        outputs.append('\t'.join(items))
    with open(filename, 'w') as fw:
        fw.write('\n'.join(outputs))

def convert_to_glue_input(instances):
    print('converting to glue input')
    head = [''] * 12
    head[0] = 'id'
    head[8] = 'claim'
    head[9] = 'evidence'
    head[-1] = 'label'
    outputs = [head]
    for instance in tqdm(instances):
        output = [''] * 12
        output[0] = instance['id']
        output[8] = instance['claim']
        output[9] = instance['evidence']
        output[-1] = instance['label']
        outputs.append(output)
    return outputs

def main(args):
    instances = read_jsonl(args.s)
    instances = process_data(instances)
    #pdb.set_trace()
    instances = convert_to_glue_input(instances)
    write_csv(args.o, instances)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', required=True, type=str)
    parser.add_argument('-o', required=True, type=str)
    args = parser.parse_args()
    print(vars(args))
    main(args)
