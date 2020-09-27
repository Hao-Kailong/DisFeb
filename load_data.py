# encoding: utf-8
import json
import torch.utils.data as data
import torch
import os
import numpy as np
import random


class FewRelDataset(data.Dataset):
    # name是文件名，root是根目录
    def __init__(self, name, encoder, N, K, Q, root):
        self.root = root
        path = os.path.join(root, name + '.json')
        if not os.path.exists(path):
            print('[ERROR] Data File Not Exist!')
            assert 0
        self.json_data = json.load(open(path, encoding='utf-8'))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],  # 句子
            item['h'][2][0],  # 实体1位置
            item['t'][2][0])  # 实体2位置
        return word, pos1, pos2, mask

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, item):
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_label = []

        for i, class_name in enumerate(target_classes):
            # 随机选择支撑集和查询集样本
            indices = np.random.choice(
                list(range(len(self.json_data[class_name]))),
                self.K + self.Q, False)
            count = 0
            for j in indices:
                # 获取一个样本
                word, pos1, pos2, mask = self.__getraw__(self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask)
                count += 1
            query_label += [i] * self.Q
        return support_set, query_set, query_label

    def  __len__(self):
        return 1000000


def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:  # 这里的k是dict.key
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label


def get_loader(name, encoder, N, K, Q, batch_size, num_workers=0, collate_fn=collate_fn, root='data'):
    dataset = FewRelDataset(name, encoder, N, K, Q, root)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)


#from encoders.sentence_encoder import BERTSentenceEncoder
#sentence_encoder = BERTSentenceEncoder(
#            'F:/Dataset/BERT/bert-base-uncased',
#            512
#        )
#loader = get_loader('demo_nyt10_hm_opennre', sentence_encoder, 5, 5, 10, batch_size=4, collate_fn=collate_fn, root='data')
#while True:
#    support, query, label = next(loader)
#    print(support)
#    print(query)
#    print(label)

