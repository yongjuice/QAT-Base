import torch
import torchvision.models as vision_models
# import horovod.torch as hvd

import numpy as np
from tqdm import tqdm

import os
import shutil
from datetime import datetime
import json
import logging
import random


class RuntimeHelper(object):
    """
        apply_fake_quantization : Flag used in layers
    """

    def __init__(self):
        self.apply_fake_quantization = False
        self.val_batch = None

        self.izero = None   ###
        self.fzero = None   ###


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_pretraining_model_checkpoint(state, is_best, path, epoch=None):
    if epoch is not None:
        filepath = os.path.join(path, 'checkpoint_{}.pth'.format(epoch))
    else:
        filepath = os.path.join(path, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(path, 'best.pth'))


def save_checkpoint(state, is_best, path):
    filepath = os.path.join(path, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(path, 'best.pth'))


def train_epoch(model, train_loader, criterion, optimizer, epoch, logger, hvd=None):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    with tqdm(train_loader, unit="batch", ncols=90) as t:
        for i, (input, target) in enumerate(t):
            t.set_description("Epoch {}".format(epoch))

            input, target = input.cuda(), target.cuda()
            output = model(input)
            loss = criterion(output, target)
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            if hvd:
                if hvd.rank() == 0:
                    logger.debug("[Epoch] {}, step {}/{} [Loss] {:.5f} (avg: {:.5f}) [Score] {:.3f} (avg: {:.3f})"
                                 .format(epoch, i + 1, len(t), loss.item(), losses.avg, prec.item(), top1.avg))
            else:
                logger.debug("[Epoch] {}, step {}/{} [Loss] {:.5f} (avg: {:.5f}) [Score] {:.3f} (avg: {:.3f})"
                             .format(epoch, i + 1, len(t), loss.item(), losses.avg, prec.item(), top1.avg))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss=losses.avg, acc=top1.avg)


def validate(model, test_loader, criterion, logger=None, hvd=None):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, unit="batch", ncols=90) as t:
            for i, (input, target) in enumerate(t):
                t.set_description("Validate")
                input, target = input.cuda(), target.cuda()
                output = model(input)
                loss = criterion(output, target)
                prec = accuracy(output, target)[0]
                losses.update(loss.item(), input.size(0))
                top1.update(prec.item(), input.size(0))

                t.set_postfix(loss=losses.avg, acc=top1.avg)

    if logger:
        if hvd:
            if hvd.rank() == 0:
                logger.debug("[Validation] Loss: {:.5f}, Score: {:.3f}".format(losses.avg, top1.avg))
        else:
            logger.debug("[Validation] Loss: {:.5f}, Score: {:.3f}".format(losses.avg, top1.avg))
    return top1.avg


def load_dnn_model(arg_dict, tools, path=None):
    model = None
    if arg_dict['quantized']:
        if arg_dict['dataset'] == 'cifar100':
            model = tools.quantized_model_initializer(arg_dict, num_classes=100)
        else:
            model = tools.quantized_model_initializer(arg_dict)
    elif arg_dict['fused']:
        if arg_dict['dataset'] == 'cifar100':
            model = tools.fused_model_initializer(arg_dict, num_classes=100)
        else:
            model = tools.fused_model_initializer(arg_dict)
    else:
        if arg_dict['dataset'] == 'imagenet':
            if arg_dict['arch'] == 'MobileNetV3':
                return vision_models.mobilenet_v3_small(pretrained=True)
            elif arg_dict['arch'] == 'ResNet18':
                return vision_models.resnet18(pretrained=True)
            elif arg_dict['arch'] == 'AlexNet':
                return vision_models.alexnet(pretrained=True)
            elif arg_dict['arch'] == 'ResNet50':
                return vision_models.resnet50(pretrained=True)
            elif arg_dict['arch'] == 'DenseNet121':
                return vision_models.densenet121(pretrained=True)
        elif arg_dict['dataset'] == 'cifar100':
            model = tools.pretrained_model_initializer(num_classes=100)
        else:
            model = tools.pretrained_model_initializer()
    if path is not None:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(arg_dict['dnn_path'])
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def load_optimizer(optim, path):
    checkpoint = torch.load(path)
    optim.load_state_dict(checkpoint['optimizer'])
    epoch_to_start = checkpoint['epoch'] + 1
    return optim, epoch_to_start


def load_tuning_info(path):
    dir_path = path.replace('/checkpoint.pth', '')
    int_params_path = os.path.join(dir_path, 'quantized/params.json')
    with open(int_params_path, 'r') as f:
        saved_args = json.load(f)
        best_epoch = saved_args['best_epoch']
        best_int_val_score = saved_args['best_int_val_score']
    return dir_path, best_epoch, best_int_val_score


def check_file_exist(path):
    return os.path.isfile(path) 


def add_path(prev_path, to_add, allow_existence=True):
    path = os.path.join(prev_path, to_add)
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if not allow_existence:
            for i in range(100):
                if not os.path.exists(path + '-{}'.format(i)):
                    path += '-{}'.format(i)
                    os.makedirs(path)
                    break
    return path


def set_save_dir(args, allow_existence=True):
    path = add_path('', 'result')
    path = add_path(path, args.mode)
    path = add_path(path, args.dataset)
    path = add_path(path, args.arch + '_' + str(args.bit) + 'bit')
    path = add_path(path, datetime.now().strftime("%m-%d-%H%M"), allow_existence=allow_existence)
    with open(os.path.join(path, "params.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    return path


def set_logger(path):
    logging.basicConfig(filename=os.path.join(path, "train.log"), level=logging.DEBUG)
    logger = logging.getLogger()
    return logger


# def metric_average(val, name):
#     tensor = torch.tensor(val)
#     avg_tensor = hvd.allreduce(tensor, name=name)
#     return avg_tensor.item()


def get_time_cost_in_string(t):
    if t > 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t > 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


def get_finetuning_model(arg_dict, tools, pretrained_model=None):
    if arg_dict['dataset'] == 'cifar100':
        fused_model = tools.fused_model_initializer(arg_dict, num_classes=100)
    else:
        fused_model = tools.fused_model_initializer(arg_dict)

    if arg_dict['fused']:
        checkpoint = torch.load(arg_dict['dnn_path'])
        fused_model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        if pretrained_model is None:
            pretrained_model = load_dnn_model(arg_dict, tools)
        fused_model = tools.fuser(fused_model, pretrained_model)
    return fused_model
