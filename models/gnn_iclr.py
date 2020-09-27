import torch
import torch.nn as nn
import torch.nn.functional as F


# 对应论文中A Matrix：求邻接矩阵
class Wcompute(nn.Module):
    def __init__(self, input_features, nf, operator='J2', activation='softmax', ratio=(2, 2, 1, 1), num_operators=1, drop=False):
        super().__init__()
        self.num_features = nf
        self.operator = operator
        self.drop = drop

        # 2 × (nf * 2)
        self.conv2d_1 = nn.Conv2d(input_features, nf * ratio[0], kernel_size=1, stride=1)
        self.bn_1 = nn.BatchNorm2d(nf * ratio[0])
        if self.drop:
            self.dropout = nn.Dropout(0.3)
        self.conv2d_2 = nn.Conv2d(nf * ratio[0], nf * ratio[1], kernel_size=1, stride=1)
        self.bn_2 = nn.BatchNorm2d(nf * ratio[1])

        # 2 × (nf)
        self.conv2d_3 = nn.Conv2d(nf * ratio[1], nf * ratio[2], kernel_size=1, stride=1)
        self.bn_3 = nn.BatchNorm2d(nf * ratio[2])
        self.conv2d_4 = nn.Conv2d(nf * ratio[2], nf, kernel_size=1, stride=1)
        self.bn_4 = nn.BatchNorm2d(nf)

        self.conv2d_last = nn.Conv2d(nf, num_operators, kernel_size=1,stride=1)
        self.activation = activation

    def forward(self, x, W_id):
        W1 = x.unsqueeze(2)
        W2 = W1.transpose(1, 2)  # (B, N, N, num_features)
        W = torch.abs(W1 - W2).transpose(1, 3)  # (B, num_features, N, N)

        W = F.leaky_relu(self.bn_1(self.conv2d_1(W)))  # (B, nf * 2, N, N)
        if self.drop:
            W = self.dropout(W)
        W = F.leaky_relu(self.bn_2(self.conv2d_2(W)))  # (B, nf * 2, N, N)
        W = F.leaky_relu(self.bn_3(self.conv2d_3(W)))  # (B, nf, N, N)
        W = F.leaky_relu(self.bn_4(self.conv2d_4(W)))  # (B, nf, N, N)

        W = self.conv2d_last(W).transpose(1, 3)  # (B, N, N, 1)

        if self.activation == 'softmax':
            W = W - W_id.expand_as(W) * torch.tensor(1e8, dtype=torch.float)
            W = W.transpose(2, 3).contiguous()
            # Applying Softmax
            W_size = W.size()
            W = W.view(-1, W.size(3))
            W = F.softmax(W)
            W = W.view(W_size).transpose(2, 3)  # 恢复形状
        elif self.activation == 'sigmoid':
            W = F.sigmoid(W)
            W *= (1 - W_id)
        elif self.activation == 'none':
            W *= (1 - W_id)
        else:
            raise NotImplementedError

        if self.operator == 'laplace':
            W = W_id - W
        elif self.operator == 'J2':
            W = torch.cat([W_id, W], dim=3)
        else:
            raise NotImplementedError

        return W


# 邻接矩阵 × 顶点矩阵
def gmul(input):
    W, x = input
    x_size = x.size()  # (B, N, num_features)
    W_size = W.size()  # (B, N, N, J)
    N = W_size[-2]
    W = W.split(1, 3)
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    return output


class Gconv(nn.Module):
    def __init__(self, nf_input, nf_output, J, bn_bool=True):
        super().__init__()
        self.J = J
        self.num_inputs = J * nf_input
        self.num_outputs = nf_output
        self.bn_bool = bn_bool

        self.fc = nn.Linear(self.num_inputs, self.num_outputs)
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, inputs):
        W = inputs[0]
        x = gmul(inputs)
        x_size = x.size()
        x = x.reshape(-1, self.num_inputs)
        x = self.fc(x)
        if self.bn_bool:
            x = self.bn(x)

        x = x.view(x_size[0], x_size[1], self.num_outputs)
        return W, x


class GNN_n1(nn.Module):
    def __init__(self, N, input_features, nf, J):
        super().__init__()
        self.input_features = input_features
        self.nf = nf
        self.J = J

        self.num_layers = 2
        self.module_w_1 = Wcompute(self.input_features, nf, operator='J2', activation='softmax', ratio=(2, 2, 1, 1))
        self.module_l_1 = Gconv(self.input_features, int(nf / 2), 2)
        self.module_w_2 = Wcompute(self.input_features + int(nf / 2), nf, operator='J2', activation='softmax', ratio=(2, 2, 1, 1))
        self.module_l_2 = Gconv(self.input_features + int(nf / 2), int(nf / 2), 2)

        self.module_w_last = Wcompute(self.input_features + int(nf / 2) * self.num_layers, nf, operator='J2', activation='softmax', ratio=(2, 2, 1, 1))
        self.module_l_last = Gconv(self.input_features + int(nf / 2) * self.num_layers, N, 2, bn_bool=False)

    def forward(self, x):
        # torch.eye()生成单位阵
        W_init = torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3).cuda()

        W0 = self.module_w_1(x, W_init)
        x_new = F.leaky_relu(self.module_l_1([W0, x])[1])
        x = torch.cat([x, x_new], dim=2)

        W1 = self.module_w_2(x, W_init)
        x_new = F.leaky_relu(self.module_l_2([W0, x])[1])
        x = torch.cat([x, x_new], dim=2)

        W = self.module_w_last(x, W_init)
        output = self.module_l_last([W, x])[1]

        return output[:, 0, :]



