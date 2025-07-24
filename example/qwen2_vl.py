from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
import codecs
from tqdm import tqdm
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="lmms-lab/llava-onevision-qwen2-7b-ov", type=str, help="path to model",)
    parser.add_argument("--device", default="auto", type=str, help="device setting")
    parser.add_argument("--img_path", default="./img", type=str, help="path to image folder",)
    parser.add_argument("--json_path", default="./qa.json", type=str, help="path to the qa.json file")
    parser.add_argument("--out_path", default="./org_results/llava_ov-7B", type=str, help="result output path")
    args = parser.parse_args()
    return args


args = get_parser()
img_dir = args.img_path
json_file = args.json_path
with open(json_file, 'r') as f:
    qa_list = json.load(f)
    f.close()

output_json_path = json_file = args.out_path
if not os.path.exists(output_json_path):
    os.mkdir(output_json_path)
if args.device != 'auto':
    device = f"cuda:{args.device}"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype=torch.bfloat16, device_map=device,  # 'auto' for 72
    attn_implementation="flash_attention_2",
)


# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=min_pixels, max_pixels=max_pixels)


results = []
results_c = []
results_h = []
results_n = []
results_p = []

for qa in tqdm(qa_list):
    question = qa['question']
    question_c = qa['question_c']
    hint = qa['hint']
    qid = qa['q_id']
    img_pth = os.path.join(img_dir, qa['img'])

    # ### CoT
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_pth,
                },
                {"type": "text", "text": f"Please answer the following question within the tags <q> and </q> step-by-step, explaining your reasoning process clearly at each stage. After presenting your reasoning, provide the final simple answer at last, making sure to enclose it within the tags <a> and </a> like this: <a>your answer</a>. \nQuestion: <q>{question}</q> \nAnswer: "},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device) # 'cuda'

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False, temperature=0, num_beams=1,)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    result = {'q_id': qid, 'answer': output_text}
    results.append(result)
    print(output_text)

    ### cross-linguistic
    messages_c = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_pth,
                },
                {"type": "text", "text": f"Please answer the following Chinese question within the tags <q> and </q> in English step-by-step, explaining your reasoning process clearly at each stage. After presenting your reasoning, provide the final simple answer at last, making sure to enclose it within the tags <a> and </a> like this: <a>your answer</a>. \nQuestion: <q>{question_c}</q> \nAnswer: "},
            ],
        }
    ]

    # Preparation for inference
    text_c = processor.apply_chat_template(
        messages_c, tokenize=False, add_generation_prompt=True
    )
    # image_inputs_c, video_inputs_c = process_vision_info(messages_c)
    inputs_c = processor(
        text=[text_c],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs_c = inputs_c.to(model.device)

    # Inference: Generation of the output
    generated_ids_c = model.generate(**inputs_c, max_new_tokens=2048, do_sample=False, temperature=0, num_beams=1,)
    generated_ids_trimmed_c = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_c.input_ids, generated_ids_c)
    ]
    output_text_c = processor.batch_decode(
        generated_ids_trimmed_c, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    result_c = {'q_id': qid, 'answer': output_text_c}
    results_c.append(result_c)
    print(output_text_c)


    ### hint
    messages_h = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_pth,
                },
                {"type": "text", "text": f"Please use the hint within the tags <h> and </h> to answer the following question within the tags <q> and </q> step-by-step, explaining your reasoning process clearly at each stage. After presenting your reasoning, provide the final simple answer at last, making sure to enclose it within the tags <a> and </a> like this: <a>your answer</a>. \nQuestion: <q>{question}</q> \nHint: <h>{hint}</h> \nAnswer: "},
            ],
        }
    ]

    # Preparation for inference
    text_h = processor.apply_chat_template(
        messages_h, tokenize=False, add_generation_prompt=True
    )
    # image_inputs_h, video_inputs_h = process_vision_info(messages_h)
    inputs_h = processor(
        text=[text_h],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs_h = inputs_h.to(model.device)

    # Inference: Generation of the output
    generated_ids_h = model.generate(**inputs_h, max_new_tokens=2048, do_sample=False, temperature=0, num_beams=1,)
    generated_ids_trimmed_h = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_h.input_ids, generated_ids_h)
    ]
    output_text_h = processor.batch_decode(
        generated_ids_trimmed_h, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    result_h = {'q_id': qid, 'answer': output_text_h}
    results_h.append(result_h)
    print(output_text_h)


    ### not CoT
    messages_n = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_pth,
                },
                {"type": "text", "text": f"Please answer the following question within the tags <q> and </q> without thinking steps. Only provide the final simple answer at last, making sure to enclose it within the tags <a> and </a> like this: <a>your answer</a>. \nQuestion: <q>{question}</q> \nAnswer: "},
            ],
        }
    ]

    # Preparation for inference
    text_n = processor.apply_chat_template(
        messages_n, tokenize=False, add_generation_prompt=True
    )
    # image_inputs_n, video_inputs_n = process_vision_info(messages_n)
    inputs_n = processor(
        text=[text_n],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs_n = inputs_n.to(model.device)

    # Inference: Generation of the output
    generated_ids_n = model.generate(**inputs_n, max_new_tokens=2048, do_sample=False, temperature=0, num_beams=1,)
    generated_ids_trimmed_n = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_n.input_ids, generated_ids_n)
    ]
    output_text_n = processor.batch_decode(
        generated_ids_trimmed_n, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    result_n = {'q_id': qid, 'answer': output_text_n}
    results_n.append(result_n)
    print(output_text_n)


    ### instruction
    messages_p = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_pth,
                },
                {"type": "text", "text": f"Please pay close attention to the textual information in the image, as well as the key elements specified in the question, such as the objects, relationships, and constraints. Then answer the following question within the tags <q> and </q> step-by-step according to the textual content, explaining your reasoning process clearly at each stage and the text clues you use. After presenting your reasoning, provide the final simple answer at last, making sure to enclose it within the tags <a> and </a> like this: <a>your answer</a>. \nQuestion: <q>{question}</q> \nAnswer: "},
            ],
        }
    ]

    # Preparation for inference
    text_p = processor.apply_chat_template(
        messages_p, tokenize=False, add_generation_prompt=True
    )
    inputs_p = processor(
        text=[text_p],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs_p = inputs_p.to(model.device)

    # Inference: Generation of the output
    generated_ids_p = model.generate(**inputs_p, max_new_tokens=2048, do_sample=False, temperature=0, num_beams=1,)
    generated_ids_trimmed_p = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_p.input_ids, generated_ids_p)
    ]
    output_text_p = processor.batch_decode(
        generated_ids_trimmed_p, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    result_p = {'q_id': qid, 'answer': output_text_p}
    results_p.append(result_p)
    print(output_text_p)

    del inputs, generated_ids, generated_ids_trimmed
    del inputs_c, generated_ids_c, generated_ids_trimmed_c
    del inputs_h, generated_ids_h, generated_ids_trimmed_h
    del inputs_n, generated_ids_n, generated_ids_trimmed_n
    del inputs_p, generated_ids_p, generated_ids_trimmed_p
    torch.cuda.empty_cache()

json_file = os.path.join(output_json_path, 'results.json')
json_fp = codecs.open(json_file, 'w', encoding='utf-8') # use codecs to speed up dump
json_str = json.dumps(results, indent=2, ensure_ascii=False)
json_fp.write(json_str)
json_fp.close()
del json_str, results

json_file_c = os.path.join(output_json_path, 'results_c.json')
json_fp_c = codecs.open(json_file_c, 'w', encoding='utf-8') # use codecs to speed up dump
json_str_c = json.dumps(results_c, indent=2, ensure_ascii=False)
json_fp_c.write(json_str_c)
json_fp_c.close()
del json_str_c, results_c

json_file_h = os.path.join(output_json_path, 'results_h.json')
json_fp_h = codecs.open(json_file_h, 'w', encoding='utf-8') # use codecs to speed up dump
json_str_h = json.dumps(results_h, indent=2, ensure_ascii=False)
json_fp_h.write(json_str_h)
json_fp_h.close()
del json_str_h, results_h

json_file_n = os.path.join(output_json_path, 'results_n.json')
json_fp_n = codecs.open(json_file_n, 'w', encoding='utf-8') # use codecs to speed up dump
json_str_n = json.dumps(results_n, indent=2, ensure_ascii=False)
json_fp_n.write(json_str_n)
json_fp_n.close()
del json_str_n, results_n

json_file_p = os.path.join(output_json_path, 'results_p.json')
json_fp_p = codecs.open(json_file_p, 'w', encoding='utf-8') # use codecs to speed up dump
json_str_p = json.dumps(results_p, indent=2, ensure_ascii=False)
json_fp_p.write(json_str_p)
json_fp_p.close()
del json_str_p, results_p