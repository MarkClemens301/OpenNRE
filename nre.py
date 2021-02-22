#!D:\ProgramData\Anaconda3\python3.6.5 (3.7.3)
# coding: utf-8
# @Author: Mark Clemens
# @Date: 2021/02/23
# @File: nre.py *
'''@notes: /TEST/ERR/OK; Scheme: '''

import opennre

print('import')
model = opennre.get_model('wiki80_cnn_softmax')

res = model.infer(
    {'text': 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).',
     'h': {'pos': (18, 46)}, 't': {'pos': (78, 91)}})
# ('father', 0.5108704566955566)

print(res)
