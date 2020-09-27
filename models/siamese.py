import sys
import torch
from torch import autograd, optim, nn


class Siamese(nn.Module):
    def __init__(self, sentence_encoder, hidden_size=230, dropout=0):
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(sentence_encoder)
        self.hidden_size = hidden_size
        self.normalize = nn.LayerNorm(normalized_shape=hidden_size)
        self.drop = nn.Dropout(dropout)
        self.cost = nn.CrossEntropyLoss()

    def forward(self, support, query, N, K, Q_total):
        '''
        :param support: Inputs of support set
        :param query: Inputs of query set
        :param N: Num of classes
        :param K: Num of instances for each class in the support set
        :param Q_total: Num of instances in the query set
        :return:
        '''

        support = self.sentence_encoder(support)  # (B * N * K, D)
        query = self.sentence_encoder(query)  # (B * Q_total, D)
        # Layer Normalization
        # 加速训练，使网络更稳定
        support = self.normalize(support)
        query = self.normalize(query)
        # Dropout
        support = self.drop(support)
        query = self.drop(query)
        support = support.view(-1, N * K, self.hidden_size)  # (B, N * K, D)
        query = query.view(-1, Q_total, self.hidden_size)  # (B, Q_total, D)
        B = support.size(0)
        support = support.unsqueeze(dim=1)  # (B, 1, N * K, D)
        query = query.unsqueeze(dim=2)  # (B, Q_total, 1, D)
        # Dot Production
        # 先对应元素相乘，后求和，即为点积
        z = (support * query).sum(dim=-1)  # (B, Q_total, N * K)
        z = z.view(-1, Q_total, N, K)  # (B, Q_total, N, K)
        # Max Combination
        # 取支撑集最大相似度的得分
        # 返回为对应结果和对应下标
        logits, _ = z.max(dim=-1)  # (B, Q_total, N)
        # NA
        minn, _ = logits.min(dim=-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, Q_total, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        return logits, pred

    def loss(self, logits, label):
        """
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        """
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))  # 计算loss

    def accuracy(self, pred, label):
        """
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        """
        return torch.mean((pred.view(-1) == label.view(-1)).float())
