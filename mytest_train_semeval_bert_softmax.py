# coding:utf-8
# import sys
# sys.path.append(r'D:\Public\Documents\GitHub\OpenNRE')

import torch
import numpy as np
import json
import opennre
from opennre import encoder, model, framework
import os

# Some basic settings
root_path = '.'
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')

# Check data
# opennre.download('semeval', root_path=root_path)
# opennre.download('bert_base_uncased', root_path=root_path)

ckpt = 'ckpt/semeval_bert_softmax.pth.tar'
rel2id = json.load(open(r'./train_data/semeval_rel2id.json'))  # , 'r', encoding='utf-8'

print(torch.cuda.is_available())
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
sentence_encoder = opennre.encoder.BERTEncoder(
    max_length=80, pretrain_path='pretrain/bert-base-uncased')

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.device_count())
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
# if torch.cuda.device_count() > 1:
#     model = torch.nn.DataParallel(model, device_ids=[0, 1])
#
#     model.to(device)

framework = opennre.framework.SentenceRE(
    train_path='./train_data/train_data_0917.json',  # txt or json 格式为json多行
    val_path='./train_data/val_data_0917.json',
    test_path='./train_data/test_data_0917.json',
    model=model,
    ckpt=ckpt,
    batch_size=1,#8
    max_epoch=2,
    lr=3e-5,
    opt='adam')
# Train
framework.train_model(metric='micro_f1')

# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
with torch.no_grad():#TODO
    result = framework.eval_model(framework.test_loader)
print('Accuracy on test set: {}'.format(result['acc']))
print('Micro Precision: {}'.format(result['micro_p']))
print('Micro Recall: {}'.format(result['micro_r']))
print('Micro F1: {}'.format(result['micro_f1']))
