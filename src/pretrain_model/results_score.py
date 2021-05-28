import numpy as np
from scorer import fever_score
import json
import re
from collections import defaultdict

ENCODING = 'utf-8'


def get_predicted_label(items):
    #labels = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    labels = ['REFUTES', 'SUPPORTS', 'NOT ENOUGH INFO']
    return labels[np.argmax(np.array(items))]


def dev_scorer(truth_file, output_file, result_file, threshold):
    fin = open(truth_file, 'rb')

    truth_list = []
    for line in fin:
        label, evidence, claim, claim_num, article, article_index, confident = line.decode(ENCODING).strip('\r\n').split('\t')
        if float(confident) >= threshold:
            truth_list.append([claim_num, label, evidence, claim, article, article_index])
    fin.close()

    fin = open(output_file, 'rb')
    lines = fin.readlines()
    results = []
    for i in range(len(lines)):
        arr = re.split(r'[\s\t]', lines[i].decode(ENCODING).strip('\r\n'))
        results.append([float(arr[0]), float(arr[1]), float(arr[2])])
    fin.close()

    claim2info = {}
    for item in truth_list:
        claim_num = int(item[0])
        if claim_num not in claim2info:
            claim2info[claim_num] = []
        claim2info[claim_num].append(item[1:])

    answers = []
    cnt = -1
    for i in range(0, 19998):
        answer = {}
        if i not in claim2info:
            answer = {"predicted_label": "NOT ENOUGH INFO",  "predicted_evidence": []}
            answers.append(answer)
            continue
        cnt += 1
        answer['predicted_label'] = get_predicted_label(results[cnt])
        answer["predicted_evidence"] = []
        for item in claim2info[i]:
            answer["predicted_evidence"].append([item[3], int(item[4])])

        answers.append(answer)
    true_answers = []
    fin = open(result_file, 'rb')
    lines = fin.readlines()
    for i in range(len(lines)):
        line = lines[i]
        true_answers.append(json.loads(line.decode(ENCODING).strip('\r\n')))
    fin.close()

    strict_score, label_accuracy, precision, recall, f1 = fever_score(answers, true_answers)
    print('DEV', strict_score, label_accuracy, precision, recall, f1)

    answers_per_class = defaultdict(list)
    for p_ans, t_ans in zip(answers, true_answers):
        ans = {}
        ans['label'] = t_ans['label']
        ans['predicted_label'] = p_ans['predicted_label']
        ans['evidence'] = t_ans['evidence']                                                           
        ans['predicted_evidence'] = p_ans['predicted_evidence']
        answers_per_class[ans['label']].append(ans)
    answers_per_class = dict(answers_per_class)
     
    for key in answers_per_class.keys():
        strict_score, label_accuracy, precision, recall, f1 = fever_score(answers_per_class[key])
        print(key, strict_score, label_accuracy, precision, recall, f1)


def test_collector(truth_file, output_file, result_file, threshold):
    fin = open(truth_file, 'rb')

    truth_list = []
    for line in fin:
        arr = line.decode(ENCODING).strip('\r\n').split('\t')
        label = arr[0]
        evidence = arr[1]
        claim = arr[2]
        claim_num = arr[3]
        article = arr[4]
        article_index = arr[5]
        confidence = float(arr[6])

        if confidence >= threshold:
            truth_list.append([claim_num, label, evidence, claim, article, article_index])
    fin.close()

    fin = open(output_file, 'rb')
    lines = fin.readlines()
    results = []
    for i in range(len(lines)):
        arr = lines[i].decode(ENCODING).strip('\r\n').split('\t')
        results.append([float(arr[0]), float(arr[1]), float(arr[2])])
    fin.close()

    claim2info = {}
    for item in truth_list:
        claim_num = int(item[0])
        if claim_num not in claim2info:
            claim2info[claim_num] = []
        claim2info[claim_num].append(item[1:])

    claim2id = {}
    fin = open(result_file, 'rb')
    lines = fin.readlines()
    for i in range(len(lines)):
        line = lines[i]
        claim2id[i] = json.loads(line)['id']
    fin.close()

    answers = []
    cnt = -1
    for i in range(0, 19998):
        answer = {}
        answer['id'] = claim2id[i]
        if i not in claim2info:
            answer = {"predicted_label": "NOT ENOUGH INFO",  "predicted_evidence": []}
            answers.append(answer)
            continue
        cnt += 1
        answer["predicted_label"] = get_predicted_label(results[cnt])
        answer["predicted_evidence"] = []
        for item in claim2info[i]:
            answer["predicted_evidence"].append([item[3], int(item[4])])

        answers.append(answer)

    fout = open('predictions.jsonl', 'wb')
    for answer in answers:
        fout.write(('%s\r\n' % json.dumps(answer)).encode(ENCODING))
    fout.close()


if __name__ == '__main__':
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument('--gear_file', type=str, default='./data/gear/bert-nli-dev-retrieve-set.tsv')
    parser.add_argument('--dev_file', type=str, required=True)
    parser.add_argument('--true_file', type=str, default='./data/fever/shared_task_dev.jsonl')
    args = parser.parse_args()

    pprint(vars(args))

    print('Dev score:')
    dev_scorer(args.gear_file, args.dev_file, args.true_file, 0)

    #print('Collect test results:')
    #test_collector('../data/bert/bert-nli-test-retrieve-set.tsv',
    #               '../outputs/gear-5evi-1layer-att-314seed-001/test-results.tsv',
    #               '../data/fever/shared_task_test.jsonl', 0)
    #print('Results can be found in predictions.jsonl')
