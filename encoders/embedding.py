# encoding: utf-8
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, word_vec_mat, max_length,
            word_embedding_dim=50, pos_embedding_dim=5):
        nn.Module.__init__(self)
        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        # word embedding
        word_vec_mat = torch.from_numpy(word_vec_mat)  # ndarray -> tensor
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0],self.word_embedding_dim,
                                           padding_idx=word_vec_mat.shape[0]-1)  # Embedding layer
        #self.word_embedding.weight.copy_(word_vec_mat)
        self.word_embedding.from_pretrained(word_vec_mat, freeze=False)  # 允许梯度更新
        # position embedding
        self.pos1_embedding = nn.Embedding(2*max_length, pos_embedding_dim, padding_idx=0)  # 学习得到位置嵌入
        self.pos2_embedding = nn.Embedding(2*max_length, pos_embedding_dim, padding_idx=0)

    def forward(self, inputs):
        word = inputs['word']
        pos1 = inputs['pos1']
        pos2 = inputs['pos2']
        x = torch.cat([self.word_embedding(word),
                       self.pos1_embedding(pos1),
                       self.pos2_embedding(pos2)], 2)  # (batch, seq_length, dim)
        return x