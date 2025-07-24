import os
import json
from glob import glob
import codecs
from tqdm import tqdm
import re
from openai import OpenAI
import argparse

def extract_answer(raw_ans):
    matches = re.findall(r'<a>(.*?)</a>', raw_ans)
    if matches:
        return matches[-1]
    else:
        return None

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--org_path", type=str, help="the original path to the answer",)
    parser.add_argument("--json_path", default="./qa.json", type=str, help="path to the qa.json file")
    parser.add_argument("--out_path", default="./org_results/llava_ov-7B", type=str, help="result output path")
    args = parser.parse_args()
    return args



args = get_parser()
qa_json = args.json_path
questions = {}
with open(qa_json) as f1:
    question_list = json.load(f1)
    f1.close
for qa in question_list:
    questions[qa['q_id']] = qa['question']

openai_key = 'your openai_key'
base = 'https://api.openai-proxy.com/v1'
client = OpenAI(api_key=openai_key, base_url=base)

json_list = glob(f'{args.org_path}/*.json')
out_dir = args.out_path
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for json_file in json_list:
    json_name = os.path.basename(json_file)
    results = []
    with open(json_file) as f:
        ans_dict = json.load(f)
        f.close()

    for a_ans in ans_dict:
        result = {}
        result['q_id'] = a_ans['q_id']
        question = questions[a_ans['q_id']]
        raw_ans = a_ans['answer']
        extr_ans = extract_answer(raw_ans)
        if extr_ans is not None and len(extr_ans) < 10:
            result['answer'] = extr_ans
        else: ### use gpt to extract answer
            response = client.chat.completions.create(model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are a professional answer extractor. Your responsibility is to read the provided raw answer and extract the most relevant and simplest answer in English based on the given question. The extracted answer should not contain <a> and </a>."},
                                {
                                    "role": "user",
                                    "content": f"Now, the given question is: {question}. \nThe raw answer is : {raw_ans}. \nYou only need to output the extracted answer. And if the question can be answered with 'Yes' or 'No', only need to output 'Yes' or 'No'."
                                }
                            ],
                            max_tokens=100,
                            temperature=0,
                        )
            extr_ans = response.choices[0].message.content
            print(f"q_id: {a_ans['q_id']}, question: {question}\n, extracted answer: {extr_ans}\n")
            result['answer'] = extr_ans
        results.append(result)
    output_json_file = os.path.join(out_dir, json_name)
    json_fp = codecs.open(output_json_file, 'w', encoding='utf-8') # use codecs to speed up dump
    json_str = json.dumps(results, indent=2, ensure_ascii=False)
    json_fp.write(json_str)
    json_fp.close()
    del json_str, results
        