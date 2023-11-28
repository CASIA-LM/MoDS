import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
import random

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

def instruct_dict(json_list):
    result_dict = {}
    for item in json_list:
        if item['instruction'] in result_dict:
            print('Exist the same instruction in this dataset!')
        else:
            result_dict[item['instruction']] = 0
    return result_dict

def instruct_category(json_file):
    category_set = []
    instruct_category_map = {}

    json_data = jload('./category.json')
    
    category_set = json_data.keys()
    
    for k, v in json_data.items():
        for ins in v:
            if ins in instruct_category_map and instruct_category_map.get(ins) == k:
                print('Category error!')
                print('category', k)
                print('instruction', ins)
            else:
                instruct_category_map[ins] = k
    
    return category_set, instruct_category_map




threshold = 0.0

quality_evaluation_file = sys.argv[1]

high_quality_file = sys.argv[2]

threshold = sys.argv[3]

quality_evaluation_list = jload(quality_evaluation_file)

all_num = len(quality_evaluation_list)
print('all number of instructions', len(quality_evaluation_list))

num_dict = {}

result_json = []

for item in quality_evaluation_list:
    upper_num = math.ceil(item['reward_score'])
    lower_num = math.floor(item['reward_score'])
    num_dict[(lower_num, upper_num)] = num_dict.get((lower_num,upper_num),0) + 1
    if float(item['reward_score']) > threshold:
        result_json.append(item)

print('The percent of each score interval:')
for k, v in num_dict.items():
    print(str(k)+'  :  '+str(v)+'  '+str(float(v)/all_num))

print('num of good case : ',len(result_json))

#jdump(result_json,result_file)
jdump(result_json,high_quality_file)

