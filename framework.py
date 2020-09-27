# encoding: utf-8
import torch
import os
from torch import optim
import sys


class FewShotREFramework:
    def __init__(self, train_data_loader, val_data_loader):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

    def __load_model__(self, ckpt):
        """
        :param ckpt: Path of the checkpoint 
        :return: Checkpoint dict
        """
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print('Successfully loaded checkpoint \'{}\''.format(checkpoint))
            return checkpoint
        else:
            raise Exception('No checkpoint found at \'{}\''.format(ckpt))

    def train(self,
              model,
              model_name,
              B, N_for_train, N, K, Q,
              learning_rate=1e-2,
              lr_step_size=3000,
              weight_decay=0.01,
              train_iter=50000,
              val_iter=100,
              val_step=2000,
              load_ckpt=None,
              save_ckpt=None,
              pytorch_optim=optim.AdamW,
              warmup=True,
              warmup_step=300,
              grad_iter=1):
        '''
        :param model: a FewShotREModel instance
        :param model_name: Name of the model
        :param B: Batch size
        :param N_for_train: Num of classes for each batch
        :param K: Num of instances for each class in the support set
        :param Q: Num of instances for each class in the query set
        :param learning_rate: Initial learning rate
        :param lr_step_size: Decay learning rate every lr_step_size steps
        :param weight_decay: Rate of decaying weight
        :param train_iter: Num of iterations of training
        :param val_iter:  Num of iterations of validating
        :param val_step: Validation every val_step steps
        '''
        print('Start training...')

        optimizer = pytorch_optim(model.parameters(),
                learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)

        start_iter = 0
        model.train()
        # Training
        best_acc = 0
        not_best_count = 0  # Stop training after several epochs without improvement
        iter_loss = 0.0
        iter_loss_dis = 0.0
        iter_right  = 0.0
        iter_right_dis = 0.0
        iter_sample = 0.0
        for it in range(start_iter, start_iter + train_iter):
            support, query, label = next(self.train_data_loader)
            if torch.cuda.is_available():
                for k in support:
                    support[k] = support[k].cuda()
                for k in query:
                    query[k] = query[k].cuda()
                label = label.cuda()
            logits, pred = model(support, query, N_for_train, K, Q * N_for_train)
            # 计算损失并反向传播
            loss = model.loss(logits, label) / float(grad_iter)
            right = model.accuracy(pred, label)
            loss.backward()

            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            iter_loss += loss.data.item()
            iter_right += right.data.item()
            iter_sample += 1
            sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%\n'.format(it+1, iter_loss/iter_sample, 100*iter_right/iter_sample))
            sys.stdout.flush()

            if (it + 1) % val_step == 0:
                acc = self.eval(model, B, N, K, Q, val_iter)
                model.train()
                if acc > best_acc:
                    sys.stdout.write('Best checkpoint')
                    sys.stdout.flush()
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_acc = acc
                iter_loss = 0.0
                iter_loss_dis = 0.0
                iter_right = 0.0
                iter_right_dis = 0.0
                iter_sample = 0.0

        print('\n#######################\n')
        print('Finish training ' + model_name)

    def eval(self,
             model,
             B, N, K, Q,
             val_iter,
             ckpt=None):
        '''
        :param model: a FewShotREModel instance
        :param B: Batch size
        :param N: Num of classes for each batch
        :param K: Num of instances for each class in the support set
        :param Q: Num of instances for each class in the query set
        :param val_iter: Num of iterations
        :param ckpt: Checkpoint path. Set as None if using current model parameters.
        :return: Accuracy
        '''
        print('')

        model.eval()
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            raise Exception('No test data loader')

        iter_right = 0.0
        iter_sample = 0.0
        with torch.no_grad():
            for it in range(val_iter):
                support, query, label = next(eval_dataset)
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    label = label.cuda()
                logits, pred = model(support, query, N, K, Q * N)

                right = model.accuracy(pred, label)
                iter_right += right.data.item()
                iter_sample += 1

                sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%\n'.format(it+1, 100*iter_right/iter_sample))
                sys.stdout.flush()
            print('')
        return iter_right / iter_sample
