import os
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc as misc

import torch
import torch.optim as optim
from utils.nadam import Nadam
from utils.n_adam import NAdam
import torch.optim.lr_scheduler as lrs
from utils.cls import CyclicLR
from utils.sgdr import *
from utils.warmup_lr import LRScheduler

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '':
            if args.save == '': args.save = now
            self.dir = '/mnt/hesy/experiment/' + args.save
        else:
            self.dir = '/mnt/hesy/experiment/' + args.load
            print('load from {}'.format(self.dir))
            if not os.path.exists(self.dir):
                args.load = ''
            else:
                self.log = torch.load(self.dir + '/map_log.pt')
                self.loss_log = torch.load(self.dir + '/loss_log.pt')
                #print('Continue from epoch {}...'.format(len(self.log)*args.test_every))
                print('Continue from epoch {}...'.format(len(self.loss_log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        # self.plot_map_rank(epoch)
        torch.save(self.log, os.path.join(self.dir, 'map_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False, end='\n'):
        print(log, end=end)
        if end != '':
            self.log_file.write(log + end)
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_map_rank(self, epoch):
        axis = np.linspace(1, epoch, self.log.size(0))
        label = 'Reid on {}'.format(self.args.data_test)
        labels = ['mAP','rank1','rank3','rank5','rank10']
        fig = plt.figure()
        plt.title(label)
        for i in range(len(labels)):
            plt.plot(axis, self.log[:, i].numpy(), label=labels[i])

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('mAP/rank')
        plt.grid(True)
        plt.savefig('{}/test_{}.jpg'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        pass

def make_optimizer(args, model):
    #trainable = filter(lambda x: x.requires_grad, model.parameters())

    #backone_params = list(map(id, model.model.module.backone.parameters() ))
    backone_params = []
    mgn_params = filter(lambda p: id(p) not in backone_params, model.parameters())
    base_params = filter(lambda p: id(p) in backone_params, model.parameters())
    train_list = [{'params': base_params},
                    {'params': mgn_params}]

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {
            'momentum': args.momentum,
            'dampening': args.dampening,
            'nesterov': args.nesterov
            }
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon,
            'amsgrad': args.amsgrad
        }
    elif args.optimizer == 'NADAM':
        optimizer_function = NAdam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {
            'eps': args.epsilon,
            'momentum': args.momentum
        }
    else:
        raise Exception()

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(train_list, **kwargs)

def make_scheduler(args, optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=args.gamma
        )
    elif args.decay_type.find('cyclic') >= 0:
        cyc_args = args.decay_type.split('_')[1:]
        base_lr = float(cyc_args[0])
        max_lr = float(cyc_args[1])
        step_size = float(cyc_args[2])
        print(base_lr, max_lr, step_size)
        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size = step_size )
    elif args.decay_type.find('sgdr') >= 0:
        cyc_args = args.decay_type.split('_')[1:]
        T_max = int(cyc_args[0])
        T_mult = int(cyc_args[1])
        scheduler = CosineAnnealingWithRestartsLR(optimizer, T_max = T_max, T_mult=T_mult)

    if args.lr_scheduler.find('warmup') >= 0:
        warmup_epoch = int(args.lr_scheduler.split('_')[1])
        warmup_begin_lr = float(args.lr_scheduler.split('_')[2])
        wu_scheduler = LRScheduler(
                        optimizer = optimizer,
                        lr_scheduler = scheduler,
                        base_lr = args.lr,
                        warmup_epoch=warmup_epoch,
                        warmup_begin_lr=warmup_begin_lr,
                        warmup_mode = 'linear'
                        )

    return wu_scheduler

