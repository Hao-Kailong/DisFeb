# encoding: utf-8
from torch import autograd, optim, nn
import torch


def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=-1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class Proto(nn.Module):
    def __init__(self, sentence_encoder, hidden_size):
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(sentence_encoder)  # 多GPU并发运行
        self.hidden_size = hidden_size
        self.drop = nn.Dropout()
        self.cost = nn.CrossEntropyLoss()

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, N, K, Q_total):
        support_emb = self.sentence_encoder(support)  # (B * N * K, D), D is the hidden size
        query_emb =self.sentence_encoder(query)  # (B * Q_total, D)

        # support = l2norm(support)
        # query = l2norm(query)

        # 对输入数据适当丢弃，强化模型
        support = self.drop(support_emb)
        query = self.drop(query_emb)
        # 调整数据形态
        support = support.view(-1, N, K, self.hidden_size)  # (B, N, K, D)
        query = query.view(-1, Q_total, self.hidden_size)  # (B, Q, D)
        B = support.size(0)
        # 计算原型
        support = torch.mean(support, 2)
        logits = -self.__batch_dist__(support, query)  # (B, Q, N)
        minn, _ = logits.min(-1)  # (B, Q)
        # 对每个元素减1
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        return logits, pred  # (B, Q, N + 1), (B * Q)

    def loss(self, logits, label):
        """
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        """
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        """
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        """
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))




