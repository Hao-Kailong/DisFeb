import torch
from torch import nn
import models.gnn_iclr as gnn_iclr
from torch.autograd import Variable


class GNN(nn.Module):
    def __init__(self, sentence_encoder, N, hidden_size=230):
        super().__init__()
        self.sentence_encoder = nn.DataParallel(sentence_encoder)
        self.hidden_size = hidden_size
        self.node_dim = hidden_size + N
        self.gnn_obj = gnn_iclr.GNN_n1(N, self.node_dim, nf=96, J=1)
        self.cost = nn.CrossEntropyLoss()

    def forward(self, support, query, N, K, Q_total):
        '''
        :param support: support set
        :param query: query set
        :param N: Num of classes
        :param K: Num of Instances for each class in the support set
        :param Q_total: Num of Instances in query set
        '''
        support = self.sentence_encoder(support)
        query = self.sentence_encoder(query)
        support = support.view(-1, N, K, self.hidden_size)
        query = query.view(-1, Q_total, self.hidden_size)

        B = support.size(0)
        D = self.hidden_size

        support = support.unsqueeze(1).expand(-1, Q_total, -1, -1, -1).reshape(-1, N * K, D)
        query = query.view(-1, 1, D)
        labels = Variable(torch.zeros((B * Q_total, N * K + 1, N), dtype=torch.float)).cuda()
        # query, C1, C2, ..., CN
        for b in range(B * Q_total):
            for i in range(N):
                for k in range(K):
                    labels[b][i * K + k + 1][i] = 1
        # 在dim上添加label信息
        nodes = torch.cat([torch.cat([query, support], dim=1), labels], dim=-1)

        logits = self.gnn_obj(nodes)  # (B * Q_total, N)
        _, pred = torch.max(logits, dim=1)
        return logits, pred

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


