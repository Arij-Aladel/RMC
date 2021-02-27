#
# created by "Arij Al Adel" (Arij.Adel@gmail.com) at 12/9/20
#


import os
import sys
import json
import torch
import random
import string
import logging
import numpy as np
import pickle as pkl
from shutil import copyfile


def load_meta(opt, meta_path):
    with open(meta_path, 'rb') as f:
        meta = pkl.load(f)
    embedding = torch.Tensor(meta['embedding'])
    opt['pos_vocab_size'] = len(meta['vocab_tag'])
    opt['ner_vocab_size'] = len(meta['vocab_ner'])
    opt['vocab_size'] = len(meta['vocab'])
    return embedding, opt, meta['vocab']


def prepare_batch_data(batch, ground_truth=True):
    batch_size = len(batch)
    # print("\nprepare :")
    # print("batch_size = len(batch)", batch_size )
    batch_dict = {}
    stat_dict = {"doc_len": [len(x['doc_tok']) for x in batch], "query_len": [len(x['query_tok']) for x in batch]}

    doc_len = max(len(x['doc_tok']) for x in batch)
    if ground_truth:
        ans_len = max(len(x['answer_tok']) for x in batch)
    # feature vector
    feature_len = len(eval(batch[0]['doc_fea'])[0]) if len(batch[0].get('doc_fea', [])) > 0 else 1
    doc_id = torch.LongTensor(batch_size, doc_len).fill_(0)
    doc_tag = torch.LongTensor(batch_size, doc_len).fill_(0)
    doc_ent = torch.LongTensor(batch_size, doc_len).fill_(0)
    doc_feature = torch.Tensor(batch_size, doc_len, feature_len).fill_(0)
    if ground_truth:
        doc_ans = torch.LongTensor(batch_size, ans_len + 2).fill_(0)

    for i, sample in enumerate(batch):
        select_len = min(len(sample['doc_tok']), doc_len)
        if select_len == 0:
            continue
        doc_id[i, :select_len] = torch.LongTensor(sample['doc_tok'][:select_len])

        if ground_truth:
            answer_tok_ori = sample['answer_tok']
            answer_tok = [2] + answer_tok_ori + [3]
            doc_ans[i, :len(answer_tok)] = torch.LongTensor(answer_tok)

    query_len = max(len(x['query_tok']) for x in batch)
    query_id = torch.LongTensor(batch_size, query_len).fill_(0)

    for i, sample in enumerate(batch):
        select_len = min(len(sample['query_tok']), query_len)
        if select_len == 0:
            continue
        query_id[i, :len(sample['query_tok'])] = torch.LongTensor(sample['query_tok'][:select_len])

    doc_mask = torch.eq(doc_id, 0)
    query_mask = torch.eq(query_id, 0)
    if ground_truth:
        ans_mask = torch.eq(doc_ans, 0)

    batch_dict['doc_tok'] = doc_id
    batch_dict['doc_pos'] = doc_tag
    batch_dict['doc_ner'] = doc_ent
    batch_dict['doc_fea'] = doc_feature
    batch_dict['query_tok'] = query_id
    batch_dict['doc_mask'] = doc_mask
    batch_dict['query_mask'] = query_mask
    if ground_truth:
        batch_dict['answer_token'] = doc_ans
        batch_dict['ans_mask'] = ans_mask

    return batch_dict, stat_dict


class BatchGen:
    def __init__(self, data_path, batch_size, gpu, is_train=True, doc_maxlen=100, query_maxlen = 101):
        self.batch_size = batch_size
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.is_train = is_train
        self.gpu = gpu
        self.data_path = data_path
        self.data = self.load(self.data_path, is_train, doc_maxlen, query_maxlen)
        if is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            data = [self.data[i] for i in indices]
        data = [self.data[i:i + batch_size] for i in range(0, len(self.data), batch_size)]
        self.data = data
        self.offset = 0

    def load(self, path, is_train, doc_maxlen=100, query_maxlen=101):
        print("query_maxlen from Batcher : ", query_maxlen)
        max = 0
        print("****************************************")
        print("doc_maxlen: ", doc_maxlen)
        print("query_maxlen: ", query_maxlen)
        print("****************************************")
        with open(path, 'r', encoding='utf-8') as reader:
            data = []
            cnt = 0
            truncated_d = 0
            truncated_q = 0
            for line in reader:
                sample = json.loads(line)
                cnt += 1
                try:
                    if len(sample['doc_tok']) > doc_maxlen:
                        sample['doc_tok'] = sample['doc_tok'][:doc_maxlen]
                        truncated_d += 1
                except TypeError:
                    print(sample['doc_tok'])
                    print(sample)
                    raise
                try:
                    if max < len(sample['query_tok']):
                        max = len(sample['query_tok'])
                    if len(sample['query_tok']) > query_maxlen:
                        sample['query_tok'] = sample['query_tok'][:query_maxlen]

                        truncated_q += 1
                except TypeError:
                    print(sample['query_tok'])
                    print(sample)
                    raise
                data.append(sample)
            print('Loaded {} samples out of {} , {}%'.format(len(data), cnt, len(data) * 100 / cnt))
            print('Truncated documents from batcher {} samples out of {}, {}%'.format(truncated_d, cnt, truncated_d*100/cnt))
            print('Truncated queries from batcher {} samples out of {}, {}%'.format(truncated_q, cnt, truncated_q*100/cnt))
            print('Max query {}'.format(max))

            return data

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
        self.offset = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]

            # Convert data into model-ready format
            batch_dict, stat_dict = prepare_batch_data(batch)

            if self.gpu:
                for k, v in batch_dict.items():
                    batch_dict[k] = v.pin_memory()
            self.offset += 1

            yield batch_dict, stat_dict
