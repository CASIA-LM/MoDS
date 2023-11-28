import dataclasses
import logging
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import io
import sys
import time
import json
import random
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

file_in = sys.argv[1]

file_out = sys.argv[2]

input_list = jload(file_in)

print('number of input file', len(input_list))

reward_name = "../models/reward-model-deberta-v3-large-v2"
rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name).cuda(), AutoTokenizer.from_pretrained(reward_name)
question, answer = "Explain nuclear fusion like I am five", "Nuclear fusion is the process by which two or more protons and neutrons combine to form a single nucleus. It is a very important process in the universe, as it is the source of energy for stars and galaxies. Nuclear fusion is also a key process in the production of energy for nuclear power plants."
inputs = tokenizer(question, answer, return_tensors='pt').to("cuda")
score = rank_model(**inputs).logits[0].detach()
print(float(score))

result_list = []
for element in input_list:
    instruction = element['instruction']
    _input = ''
    if 'input' in element.keys():
        _input = element['input']
    _output = element['output']
    question = ''
    if _input == '':
        question = instruction
    else:
        question = instruction + '\n' +_input
    
    answer = _output
    
    try:
        inputs = tokenizer(question, answer, return_tensors='pt').to("cuda")
        score = rank_model(**inputs).logits[0].detach()
    except:
        print(instruction)
        print(_output)
        continue
    final_result = {'instruction':instruction,'input':_input,'output':_output,'reward_score':float(score)}
    result_list.append(final_result)

print('number of data', len(result_list))

jdump(result_list,file_out)

