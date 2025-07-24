import re
import json
from glob import glob
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ans_path", type=str, help="path to the answer folder",)
    parser.add_argument("--json_path", default="./qa.json", type=str, help="path to the qa.json file")
    args = parser.parse_args()
    return args


def clean_answer(ans):
    ans = ans.lower()
    if 'the' in ans:
        ans = ans.replace('the ', '')

    if ans.endswith('%.'):
        ans = ans.rstrip('%.')
    elif ans.endswith('%'):
        ans = ans.rstrip('%')
    elif ans.endswith('.'):
        ans = ans.rstrip('.')
    return ans


args = get_parser()
gt_json = args.json_path
with open(gt_json) as f1:
    gts = json.load(f1)
    f1.close

gt_ans = {}
for gt in gts:
    gt_ans[gt['q_id']] = gt['answer']

test_jsons = glob(f'{args.ans_path}/*.json')

acc_total = {'CoT': 0, 'not CoT': 0, 'Cross language': 0, 'Hint': 0, 'Instruction': 0, }
for test_json in test_jsons:
    with open(test_json) as f:
        test_ans = json.load(f)
        f.close

    correct = 0
    for extr_ans in test_ans:
        qid = extr_ans['q_id']
        gt = clean_answer(gt_ans[qid])
        ans = clean_answer(extr_ans['answer'])
        if gt == ans:
            correct += 1
        else:
            print(f"qid: {qid}, gt: {gt}, ans: {ans}")
    acc = round(correct * 100.0 / len(test_ans), 1)
    if 'results_n' in test_json:
        acc_total['not CoT'] = acc
    elif 'results_c' in test_json:
        acc_total['Cross language'] = acc
    elif 'results_h' in test_json:
        acc_total['Hint'] = acc
    elif 'results_p' in test_json:
        acc_total['Instruction'] = acc
    elif 'results.' in test_json:
        acc_total['CoT'] = acc
    else:
        continue
    print(test_json)
    print(f"num: {len(test_ans)}, correct: {correct}, acc: {acc}")
print(acc_total)