import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import json
import sys
import numpy as np
from transformers import BertTokenizer, BertModel,AutoModel
import torch
from kcenter_greedy import *


@torch.no_grad()
def bert_embedding(texts,batch=100):
    tokenizer = BertTokenizer.from_pretrained('../models/bert-base-uncased')
    model = AutoModel.from_pretrained('../models/bert-base-uncased').cuda()
    # 将文本转化为BERT模型可识别的token序列
    encoded_texts = tokenizer(texts,return_tensors="pt",truncation=True,padding=True,max_length=96)
    encoded_texts =  encoded_texts.to("cuda")
    cls_hid_li = []
    # 使用BERT模型对每个文本序列进行编码,提取其语义向量
    i= 0
    while i < len(texts):
        last_hids = model(input_ids=encoded_texts["input_ids"][i:i+batch],
                          attention_mask=encoded_texts["attention_mask"][i:i+batch])['last_hidden_state']
        cls_hids = last_hids[:,0,:].squeeze()
        cls_hid_li.append(cls_hids)
        i+= batch
        print(i)
    # 将所有文本的embedding连成特征矩阵
    cls_hids_tensor = torch.concat(cls_hid_li, dim=0)
    np.save("bert_embedding.npy",cls_hids_tensor.cpu())
    return np.array(cls_hids_tensor.cpu())

# 数据采样
def sample_func(text_list,K):
    result = []
    if os.path.exists("bert_embedding.npy"):
        text_embedding = np.load("bert_embedding.npy")
    else:
        text_embedding = bert_embedding(text_list)
        np.save("bert_embedding.npy",text_embedding)
    
    result = []

    k_center = kCenterGreedy(text_embedding)
    
    already_selected = None
    #for _ in range(K):
    result = k_center.select_batch_(text_embedding,already_selected,K)
        #result = result + new_data
        #already_selected += new_data
    return result


def main(input_file, output_file, K):
    data = json.load(fp=open(input_file, "r"))
    instruction_list = []
    for d in data:
        instruction_list.append(d["instruction"])
    res = sample_func(text_list = instruction_list, K = K)
    print('data length')
    print(len(data))
    
    print('sampling data:')
    print(len(res))
    print(res)
    data_li = []
    for index in res:
        data_li.append(data[index])
    json.dump(obj=data_li,fp=open(output_file,"w"),indent=2,ensure_ascii=False)

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    K = int(sys.argv[3])
    main(input_file, output_file, K)
