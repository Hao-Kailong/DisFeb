from torch import nn
import numpy as np
import torch
import sys
# import torch.nn.functional as F  # Deprecated


# 因果卷积，哪里体现了因果呢？dilation吗？
# 因为是逐层递推的，所以是因果卷积
# 假设seq=8, 则layer=0,1,2, 经过三层CausalConv单词维度划分为：
# [embedding0, 0,  0,    0]
# [embedding3, 3,  23,   0123]
# [embedding7, 67, 4567, 01234567]
# 不断汇聚前文信息


class CausalConv1d(nn.Module):
    # dilation: 膨胀
    # kernel_size始终为2
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=2):
        super().__init__()
        self.padding = dilation
        self.causal_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, minibatch):
        # 保持序列长度不变
        return self.causal_conv(minibatch)[:, :, :-self.padding]


class DenseBlock(nn.Module):
    """卷积后拼接在一起"""
    def __init__(self, in_channels, filters, dilation=2):
        super().__init__()
        self.causal_conv1 = CausalConv1d(in_channels, out_channels=filters, dilation=dilation)
        self.causal_conv2 = CausalConv1d(in_channels, out_channels=filters, dilation=dilation)

    def forward(self, minibatch):
        tanh = torch.tanh(self.causal_conv1(minibatch))
        sig = torch.sigmoid(self.causal_conv2(minibatch))
        # element-wise product
        return torch.cat([minibatch, tanh * sig], dim=1)


class TCBlock(nn.Module):
    def __init__(self, in_channels, filters, seq_len):
        """
        不能处理变长数据，可能需要save then load
        """
        super().__init__()
        layer_count = np.ceil(np.log2(seq_len)).astype(np.int)  # 取对数，上取整
        blocks = []
        channel_count = in_channels
        for layer in range(layer_count):
            # 序列长度：
            # L_out = L_in + 2 * padding - dilation
            # 经过DenseBlock序列长度是不变的
            block = DenseBlock(channel_count, filters, dilation=2**layer)
            blocks.append(block)
            channel_count += filters
        self.tcblock = nn.Sequential(*blocks)  
        self._dim = channel_count

    def forward(self, minibatch):
        return self.tcblock(minibatch)

    @property
    def dim(self):  # 向量维度
        return self._dim


class AttentionBlock(nn.Module):
    def __init__(self, dims, k_size, v_size):
        """
        self-attention
        :param dims: 输入维度
        :param k_size: key维度
        :param v_size: value维度
        """
        super().__init__()
        self.key_layer = nn.Linear(dims, k_size)
        self.query_layer = nn.Linear(dims, k_size)
        self.value_layer = nn.Linear(dims, v_size)
        self.sqrt_k = np.sqrt(k_size)
        # np.tril()，下三角阵
        mask = np.tril(np.ones((1000, 1000))).astype(np.float)  # 序列长度不允许超过1000
        self.mask = nn.Parameter(torch.from_numpy(mask), requires_grad=False)
        self.minus = -100.
        self._dim = dims + v_size

    def forward(self, minibatch, seq_len):
        keys = self.key_layer(minibatch)
        # queries = self.query_layer(minibatch)
        queries = keys.clone()  # self-attention Vaswani et al. (2017a)
        values = self.value_layer(minibatch)
        current_mask = self.mask[:seq_len, :seq_len]
        # bmm: batch matrix-matrix product
        # 缩放点积模式
        # 下三角保持不变，上三角全为-100
        # 遮蔽住后面的，只能看到自己和自己前面的......
        logits = torch.div(torch.bmm(queries, keys.transpose(2, 1)), self.sqrt_k)
        logits *= current_mask
        logits += self.minus * (1. - current_mask)
        probs = torch.softmax(logits, dim=2)  # 理解：-100转化为概率很小，相当于为零
        # 按照相似度得到当前word嵌入上下文信息的表示
        read = torch.bmm(probs.float(), values)
        # 将得到的v_size的representation拼接到向量维度上
        return torch.cat([minibatch, read], dim=2)

    # 将方法转化为相同名称的只读属性
    @property
    def dim(self):
        return self._dim


class SNAIL(nn.Module):
    def __init__(self, sentence_encoder, N, K, hidden_size):
        """
        :param sentence_encoder:
        :param N: num of classes
        :param K: num of instances
        :param hidden_size:
        """
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(sentence_encoder)
        self.hidden_size = hidden_size
        self.drop = nn.Dropout()
        self.seq_len = N * K + 1  # 参见forward
        self.cost = nn.CrossEntropyLoss()

        self.att0 = AttentionBlock(hidden_size + N, k_size=64, v_size=32)
        self.tc1 = TCBlock(self.att0.dim, filters=128, seq_len=self.seq_len)
        self.att1 = AttentionBlock(self.tc1.dim, 256, 128)
        self.tc2 = TCBlock(self.att1.dim, 128, self.seq_len)
        self.att2 = AttentionBlock(self.tc2.dim, 512, 256)
        # 直接一个全连接作分类
        self.linear = nn.Linear(self.att2.dim, out_features=N, bias=False)
        # 规范化层
        self.bn1 = nn.BatchNorm1d(self.tc1.dim)
        self.bn2 = nn.BatchNorm1d(self.tc2.dim)

    def forward(self, support, query, N, K, Q_total):
        support = self.sentence_encoder(support)  # (B * N * K, D)
        query = self.sentence_encoder(query)  # (B * Q_total, D)
        # 增强数据
        # support = self.drop(support)
        # query = self.drop(query)
        support = support.view(-1, N, K, self.hidden_size)  # (B, N, K, D)
        query = query.view(-1, Q_total, self.hidden_size)  # (B, Q_total, D)
        B = support.size(0)
        Q_total = query.size(1)
        support = support.unsqueeze(1).expand(-1, Q_total, -1, -1, -1).reshape(
            -1, N * K, self.hidden_size)  # (B * Q_total, N * K, D)
        query = query.view(-1, 1, self.hidden_size)  # (B * Q_total, 1, D)
        # 前 N * K 为support sample，最后一个为query sample
        minibatch = torch.cat([support, query], 1)  # (B * Q_total, N * K + 1, D)
        labels = torch.zeros((B * Q_total, N * K + 1, N)).float().cuda()
        minibatch = torch.cat((minibatch, labels), 2)
        for i in range(N):
            for j in range(K):
                minibatch[:, i * K + j, i] = 1
        # 注意力层
        x = self.att0(minibatch, self.seq_len)  # (B * Q_total, N * K + 1, D + v_size)
        # 投入Conv1d前需要转换维度
        x = self.bn1(self.tc1(x.transpose(1, 2)))
        x = self.att1(x.transpose(1, 2), self.seq_len)
        x = self.bn2(self.tc2(x.transpose(1, 2)))
        x = self.att2(x.transpose(1, 2), self.seq_len)
        # 只保留序列的最后一个元素，即query，汇聚了所有的信息
        x = x[:, -1, :]
        logits = self.linear(x)  # (B * Q_total, N)
        _, pred = torch.max(logits, -1)  # 这里没有拼接，和其他处理不太一样
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

