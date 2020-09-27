# encoding: utf-8
import argparse
from load_data import *
from torch import optim
from models.proto import Proto
from models.siamese import Siamese
from models.snail import SNAIL
from models.gnn import GNN
from models.metanet import MetaNet
import numpy as np
from encoders.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder
from framework import FewShotREFramework
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train_wiki',
            help='train file')
    parser.add_argument('--val', default='val_wiki',
            help='validation file')
    parser.add_argument('--test', default='demo_nyt10_hm_opennre',
            help='test file')
    parser.add_argument('--trainN', default=10, type=int,
            help='N in train')
    parser.add_argument('--N', default=5, type=int,
            help='N way')
    parser.add_argument('--K', default=5, type=int,
            help='K shot')
    parser.add_argument('--Q', default=10, type=int,
            help='num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=50000, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=100, type=int,
            help='num of iters in validation')
    parser.add_argument('--val_step', default=2000, type=int,
            help='val after training how many iters')
    parser.add_argument('--model', default='proto',
            help='model name')
    parser.add_argument('--encoder', default='bert',
            help='encoder: cnn or bert or lstm')
    parser.add_argument('--max_length', default=128, type=int,
            help='max length')
    parser.add_argument('--lr', default=1e-2, type=float,
            help='learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float,
            help='weight decay')
    parser.add_argument('--dropout', default=0.1, type=float,
            help='dropout rate')
    parser.add_argument('--grad_iter', default=1, type=int,
            help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='adam',
            help='[sgd, adam]')
    parser.add_argument('--hidden_size', default=230, type=int,
            help='hidden size')
    parser.add_argument('--load_ckpt', default=None,
            help='load checkpoint')
    parser.add_argument('--save_ckpt', default=None,
            help='save checkpoint')

    opt = parser.parse_args()
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length

    print('{}-way-{}-shot Few-Shot Relation Classification'.format(N, K))
    print('model: {}'.format(model_name))
    print('encoder: {}'.format(encoder_name))
    print('max_length: {}'.format(max_length))

    if encoder_name == 'cnn':
        try:
            glove_mat = np.load('F:/Dataset/Glove/glove_mat.npy')
            glove_word2id = json.load(open('F:/Dataset/Glove/glove_word2id.json'))
        except:
            raise Exception('Can not find glove files. Run glove/download_glove.sh to download glove files')
        sentence_encoder = CNNSentenceEncoder(
            glove_mat,
            glove_word2id,
            max_length
        )
    elif encoder_name == 'bert':
        sentence_encoder = BERTSentenceEncoder(
            'F:/Dataset/BERT/bert-base-uncased',
            max_length
        )
        opt.hidden_size = 768
    else:
        raise NotImplementedError

    train_data_loader = get_loader(opt.train, sentence_encoder,
            N=trainN, K=K, Q=Q, batch_size=batch_size)
    val_data_loder = get_loader(opt.val, sentence_encoder,
            N=N, K=K, Q=Q, batch_size=batch_size)

    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, opt.test, str(N), str(K)])

    if model_name == 'proto':
        model = Proto(sentence_encoder, hidden_size=opt.hidden_size)
    elif model_name == 'snail':
        model = SNAIL(sentence_encoder, N, K, hidden_size=opt.hidden_size)
    elif model_name == 'siamese':
        model = Siamese(sentence_encoder, opt.hidden_size)
    elif model_name == 'gnn':
        model = GNN(sentence_encoder, N, opt.hidden_size)
    elif model_name == 'metanet':
        model = MetaNet(N, K, glove_mat, opt.max_length, opt.hidden_size)
    else:
        raise NotImplementedError

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.AdamW
    else:
        raise NotImplementedError

    # 模型可视化
    # with SummaryWriter(comment=model_name) as w:
    #    w.add_graph(model, (torch.tensor(np.random.rand(batch_size * N * K, 230)),
    #            torch.tensor(np.random.rand(batch_size * N * Q, 230)),
    #            torch.tensor(N),
    #            torch.tensor(K),
    #            torch.tensor(N * Q)))

    framework = FewShotREFramework(train_data_loader, val_data_loder)
    framework.train(model, prefix, batch_size, trainN, N, K, Q,
            pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
            val_step=opt.val_step, train_iter= opt.train_iter, val_iter=opt.val_iter)
    # 测试阶段
    acc = framework.eval(model, batch_size, N, K, Q, opt.val_iter, ckpt=ckpt)
    print('RESULT: {:.2f}'.format(acc * 100))


if __name__ == '__main__':
    main()

