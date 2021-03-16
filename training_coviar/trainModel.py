"""Run training."""
import logging
import os
import shutil
import time
import numpy as np
import pandas as pd

import torch
from torch import distributions
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import torch.nn.parallel
import torchvision

from training_coviar.dataset import CoviarDataSet, CoviarTestDataSet
from training_coviar.coviarModel import Model
from train_options import parser
from training_coviar.coviarTransforms import GroupCenterCrop
from training_coviar.coviarTransforms import GroupScale

import config_logger
from utilities import COVIAR_DATAFRAME_PATH, MODELS_DIECTORY

log = logging.getLogger(__name__)

PRINT_FREQ = 250

model_save_path = os.path.join(MODELS_DIECTORY, "coviar_model")


def train(train_loader, model, criterion, optimizer, epoch, cur_lr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()

    for i, pairs in enumerate(train_loader):
        data_time.update(time.time() - end)

        real_input = pairs["real"].to(gpu)
        fake_input = pairs["fake"].to(gpu)

        target = probability_distribution.sample((len(fake_input),)).float().to(gpu)
        fake_weight = target.view(-1, 1, 1, 1)

        input_tensor = (1.0 - fake_weight) * real_input + fake_weight * fake_input

        # zero the parameter gradients
        optimizer.zero_grad()

        output = model(input_tensor)
        print(output.shape)
        output = output.view((-1, args.num_segments) + output.size()[1:])
        print(output.shape)

        loss = criterion(output, target)

        # Might leave, but maybe not (come back) need to debug to make sure correct
        losses.update(loss.item(), input_tensor.shape[0])

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            log.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                            epoch, i, len(train_loader),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            lr=cur_lr)))


def evaluate(model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    end = time.time()
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(gpu)
        labels = labels.to(gpu)

        outputs = model(inputs)
        print(outputs.shape)
        # output = output.view((-1, args.num_segments) + output.size()[1:])
        # output = torch.mean(output, dim=1)
        y_pred = outputs.squeeze()
        labels = labels.type_as(y_pred)
        loss = criterion(y_pred, labels)

        losses.update(loss.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            log.info(('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(test_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses)))

    return losses.avg


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


def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay):
    decay = lr_decay ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    wd = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = wd * param_group['decay_mult']
    return lr


if __name__ == '__main__':
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = {
        "train": [],
        "val": []
    }
    minimum_loss = 100
    log.info("Program Started")
    log.info(f"GPU value: {gpu}")
    args = parser.parse_args()

    probability_distribution = distributions.beta.Beta(0.5, 0.5)

    log.info('Training arguments:')
    for k, v in vars(args).items():
        log.info('\t{}: {}'.format(k, v))

    num_class = 1

    model = Model(num_class, args.num_segments, args.representation,
                  base_model=args.arch)

    log.info("Model is prepared")

    training_dataframe = pd.read_csv(COVIAR_DATAFRAME_PATH)
    testing_dataframe = pd.read_csv(COVIAR_DATAFRAME_PATH)

    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            training_dataframe,
            num_segments=args.num_segments,
            representation=args.representation,
            transform=model.get_augmentation(),
            accumulate=(not args.no_accumulation),
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    log.info("Train Loader is prepared")

    test_loader = torch.utils.data.DataLoader(
        CoviarTestDataSet(
            testing_dataframe,
            num_segments=args.num_segments,
            representation=args.representation,
            transform=torchvision.transforms.Compose([
                GroupScale(int(model.scale_size)),
                GroupCenterCrop(model.crop_size),
            ]),
            accumulate=(not args.no_accumulation),
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    cudnn.benchmark = True

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key else 1.0

        if ('module.base_model.conv1' in key
            or 'module.base_model.bn1' in key
            or 'data_bn' in key) and args.representation in ['mv', 'residual']:
            lr_mult = 0.1
        elif '.fc.' in key:
            lr_mult = 1.0
        else:
            lr_mult = 0.01

        params += [{'params': value, 'lr': args.lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult}]

    optimizer = torch.optim.Adam(
        params,
        weight_decay=args.weight_decay,
        eps=0.001)
    criterion = F.binary_cross_entropy_with_logits

    for epoch in range(args.epochs):
        cur_lr = adjust_learning_rate(optimizer, epoch, args.lr_steps, args.lr_decay)

        train(train_loader, model, criterion, optimizer, epoch, cur_lr)

        testing_loss = evaluate(model, minimum_loss)
        history["test"].append(testing_loss)

        log.info(history)

        # deep copy the model
        if testing_loss < minimum_loss:
            minimum_loss = testing_loss
            log.info("Minimum loss is: {}".format(minimum_loss))
            path_model = model_save_path + str(epoch) + ".pth"
            torch.save(model.state_dict(), path_model)
