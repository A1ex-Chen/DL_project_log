# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

Pretraining of ResNet with exchanged encoder blocks on ImageNet. This script
uses Tensorflow Datasets, as the dataset is already available as tfrecords.
Part of this code is copied from:
https://github.com/pytorch/examples/blob/master/imagenet/main.py

"""
import os
import json
import time
import argparse
import torch
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
tf.compat.v1.enable_eager_execution()
import tensorflow_datasets as tfds

from src.models.resnet import ResNet34, ResNet18
from src.logger import CSVLogger



















class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'











class ProgressMeter(object):




    model = Classifier()
    print(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Device:', device)
    model.to(device)

    return model, device


def train(train_batches, model, criterion, optimizer, epoch, device,
          n_train_images, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        n_train_images,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for i, train_batch in enumerate(train_batches):
        images = torch.from_numpy(train_batch[0].numpy())
        target = torch.from_numpy(train_batch[1].numpy())

        # do not train on the last smaller batch
        current_batch_size = len(target)
        if current_batch_size < args.batch_size:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display((i+1)*args.batch_size)

    logs = dict()
    logs['acc_train_top-1'] = top1.avg.cpu().numpy().item()
    logs['acc_train_top-5'] = top5.avg.cpu().numpy().item()
    logs['loss_train'] = losses.avg
    return logs


def validate(validation_batches, model, criterion, device, n_val_images, logs,
             args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        n_val_images,
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        examples_done = 0
        for i, validation_batch in enumerate(validation_batches):
            images = torch.from_numpy(validation_batch[0].numpy())
            target = torch.from_numpy(validation_batch[1].numpy())

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            examples_done += len(target)
            if i % args.print_freq == 0:
                progress.display(examples_done)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5), flush=True)
        logs['acc_val_top-1'] = top1.avg.cpu().numpy().item()
        logs['acc_val_top-5'] = top5.avg.cpu().numpy().item()
        logs['loss_val'] = losses.avg

    return logs


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_ckpt(ckpt_dir, model, optimizer, epoch, is_best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    # save best checkpoint with epoch number
    if is_best:
        ckpt_model_filename = "ckpt_epoch_{}.pth".format(epoch)
        path = os.path.join(ckpt_dir, ckpt_model_filename)
        torch.save(state, path)
        print('{:>2} has been successfully saved'.format(path), flush=True)
    # always save latest checkpoint
    torch.save(state, os.path.join(ckpt_dir, 'ckpt_latest.pth'))


if __name__ == '__main__':
    main()