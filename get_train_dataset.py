#!/usr/bin/env python3.7
# -*- coding: UTF-8 -*-
"""
# @Company ：华中科技大学机械学院数控中心
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/8/12 22:21
# @File    : get_train_dataset.py
# @Software: PyCharm
"""

# -*- coding: utf-8 -*-

import os
import numpy as np
# import thulac
import random
import sys

sys.path.append("..")
# from toolkit.pre_load import pre_load_thu,neo_con,predict_labels
# from toolkit.NER import get_NE,temporaryok,get_explain,get_detail_explain
import json

'''
mark备注：
'''

'''
Inputs:
"en_relations_0820.csv"  # TODO Macintoshi CR
"entities_match_ch_en_0820.csv"

Outputs:
train_data_2w_0917_1_.txt >>
train_data_0917_.json
val_data_0917_.json
test_data_0917_.json
'''

# 分句标识符号
stopToken = ".!?"


def CutStatements(line):
    statements = []
    tokens = []
    for token in line:
        tokens.append(token)
        # 如果是句子停止词
        if (token in stopToken):
            statements.append(''.join(tokens))
            tokens = []
    if (len(tokens) > 2):  # 如果只有一句
        statements.append(''.join(tokens) + "。")
    return statements


entities1_entities2 = []
with open("en_relations_0820.csv", 'r', encoding='utf-8') as f:
    for n in f.readlines():
        lines = n.strip().lower().split(",")
        entities1_entities2.append(lines)
entities = []
with open("entities_match_ch_en_0820.csv", 'r', encoding='utf-8') as f:
    for n in f.readlines():
        lines = n.strip().lower().split(",")
        entities.append(lines[1])
print('实体数和关系对数', len(entities), len(entities1_entities2))  #

# 生成负样本
f_entities1_entities2 = []
for n in entities:
    entities_f_2 = random.sample(entities, 5)
    for k in entities_f_2:
        temp = [n, k]
        if temp not in entities1_entities2:  # save不存在的实体关系
            f_entities1_entities2.append(temp)
print('(生成的负样本)关系对数', len(f_entities1_entities2))

ner_id = 1001
ner_dict_new = {}  # 存储所有实体统一后结果与其编号对应关系
train_list = []

'''
{"text": "an industrial robot and a method for controlling an industrial robot.", 
"relation": "is_subclass_of", 
"h": {"name": "industrial robot", "pos": [3, 19], "id": "entity_1001"}, 
"t": {"name": "robot", "pos": [14, 19], "id": "entity_1002"}}
'''
with open("train_data/train_data_2w_0917_1_.txt", 'w') as fw:
    # fw.write('entity1Pos\tentity1\tentity2Pos\tentity2\tstatement\trelation\n')
    with open("标题和摘要.txt", 'r', encoding='utf-8') as fr:
        count = 0
        for line in fr:
            count += 1
            if (count % 10000 == 0):
                print("标题和摘要.txt" + "  " + str(count))
            # 过滤掉<doc >  </doc> 等无用行
            if (len(line) < 2 or line[0:4] == '<doc' or line[0:6] == "</doc>"):
                continue
            # 分句
            statements = CutStatements(line)
            for statement in statements:
                # 分词
                for n in entities1_entities2:
                    entity_1, entity_2 = n
                    statement = statement.lower()
                    if entity_1 in statement and entity_2 in statement and entity_1.strip() != entity_2.strip():
                        entity_1_pos_start = statement.index(entity_1, 0)
                        entity_1_pos_end = entity_1_pos_start + len(entity_1)
                        entity_1_pos = [entity_1_pos_start, entity_1_pos_end]
                        # entity_1_pos = '[' + str(entity_1_pos_start) + ' ,' + str(entity_1_pos_end) + ']'
                        entity_2_pos_start = statement.index(entity_2, 0)
                        entity_2_pos_end = entity_2_pos_start + len(entity_2)
                        entity_2_pos = [entity_2_pos_start, entity_2_pos_end]
                        # entity_2_pos = '[' + str(entity_2_pos_start) + ' ,' + str(entity_2_pos_end) + ']'
                        train_data = dict()
                        h = dict()
                        t = dict()
                        train_data['text'] = statement.strip()
                        train_data['relation'] = "is_subclass_of"
                        h["name"] = entity_1.strip()
                        h["pos"] = entity_1_pos
                        if entity_1.strip() not in ner_dict_new:
                            ner_dict_new[entity_1.strip()] = ner_id
                            ner_id += 1
                        h['id'] = 'entity_' + str(ner_dict_new[entity_1.strip()])
                        train_data['h'] = h
                        t["name"] = entity_2.strip()
                        t["pos"] = entity_2_pos
                        if entity_2.strip() not in ner_dict_new:
                            ner_dict_new[entity_2.strip()] = ner_id
                            ner_id += 1
                        t['id'] = 'entity_' + str(ner_dict_new[entity_2.strip()])
                        train_data['t'] = t

                        train_data_json = json.dumps(train_data, ensure_ascii=False)
                        train_list.append(train_data)
                        fw.write(train_data_json + "\n")
                        # if entity_1_pos_start != entity_2_pos_start and entity_1_pos_end != entity_2_pos_end and \
                        #         entity_2_pos_end != entity_1_pos_end - 1:
                        #     fw.write(str(entity_1_pos)+'\t'+entity_1+'\t' + str(entity_2_pos)
                        #              +'\t'+entity_2+'\t'+statement.strip()+'\t'+"is_subclass_of"+'\n')

                for n in f_entities1_entities2:
                    entity_1, entity_2 = n
                    statement = statement.lower()
                    if entity_1 in statement and entity_2 in statement and entity_1.strip() != entity_2.strip():
                        entity_1_pos_start = statement.index(entity_1, 0)
                        entity_1_pos_end = entity_1_pos_start + len(entity_1)
                        entity_1_pos = [entity_1_pos_start, entity_1_pos_end]
                        # entity_1_pos = '[' + str(entity_1_pos_start) + ' ,' + str(entity_1_pos_end) + ']'
                        entity_2_pos_start = statement.index(entity_2, 0)
                        entity_2_pos_end = entity_2_pos_start + len(entity_2)
                        entity_2_pos = [entity_2_pos_start, entity_2_pos_end]
                        # entity_2_pos = '[' + str(entity_2_pos_start) + ' ,' + str(entity_2_pos_end) + ']'
                        train_data = dict()
                        h = dict()
                        t = dict()
                        train_data['text'] = statement.strip()
                        train_data['relation'] = "NA"
                        h["name"] = entity_1.strip()
                        h["pos"] = entity_1_pos
                        if entity_1.strip() not in ner_dict_new:
                            ner_dict_new[entity_1.strip()] = ner_id
                            ner_id += 1
                        h['id'] = 'entity_' + str(ner_dict_new[entity_1.strip()])
                        train_data['h'] = h
                        t["name"] = entity_2.strip()
                        t["pos"] = entity_2_pos
                        if entity_2.strip() not in ner_dict_new:
                            ner_dict_new[entity_2.strip()] = ner_id
                            ner_id += 1
                        t['id'] = 'entity_' + str(ner_dict_new[entity_2.strip()])
                        train_data['t'] = t
                        train_list.append(train_data)
                        train_data_json = json.dumps(train_data, ensure_ascii=False)
                        fw.write(train_data_json + "\n")

print('数据集大小', len(train_list))
np.random.shuffle(train_list)  # 训练集7：验证集1：测试集2
num = int(len(train_list) * 0.7)
num_v = int(len(train_list) * 0.1)
num_t = num + num_v
train, val, test = train_list[:num], train_list[num: num_t], train_list[num_t:]

with open("train_data/train_data_0917_.json", 'a') as fw:  # why a-mode used?
    for n in train:
        train_data_json = json.dumps(n, ensure_ascii=False)
        fw.write(train_data_json + "\n")

with open("train_data/val_data_0917_.json", 'a') as fw:
    for n in val:
        train_data_json = json.dumps(n, ensure_ascii=False)
        fw.write(train_data_json + "\n")

with open("train_data/test_data_0917_.json", 'a') as fw:
    for n in test:
        train_data_json = json.dumps(n, ensure_ascii=False)
        fw.write(train_data_json + "\n")
