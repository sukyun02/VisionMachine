# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for adjusting keep rate -- Youwei Liang
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from losses import DistillationLoss
import utils
from helpers import adjust_keep_rate

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    writer=None,
                    set_training_mode=True,
                    args=None,
                    sc_smooth_matrix=None):


    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # p.s. better to use 20 for cifar and 200 for imagenet
    print_freq = 20
    log_interval = 10
    it = epoch * len(data_loader)
    ITERS_PER_EPOCH = len(data_loader)

    accum_steps = getattr(args, 'accum_steps', 1)
    scaler = loss_scaler._scaler  # underlying GradScaler

    for batch_idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            if sc_smooth_matrix is not None:
                # targets is lam * one_hot(t1) + (1-lam) * one_hot(t2) (smoothing=0)
                # Multiply by smooth_matrix to get superclass-aware soft targets
                targets = targets @ sc_smooth_matrix
        elif sc_smooth_matrix is not None:
            # No mixup: convert hard labels to sc-aware soft targets
            targets = sc_smooth_matrix[targets]

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)
            loss = loss / accum_steps

        loss_value = loss.item() * accum_steps  # log unscaled loss

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        scaler.scale(loss).backward(create_graph=is_second_order)

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(data_loader):
            if max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            torch.cuda.synchronize()
            if model_ema is not None:
                model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if torch.distributed.get_rank() == 0 and it % log_interval == 0:
            writer.add_scalar('loss', loss_value, it)
            writer.add_scalar('lr', optimizer.param_groups[0]["lr"], it)
        it += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, keep_rate=None, fine_to_coarse=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Prepare superclass lookup tensor
    if fine_to_coarse is not None:
        fine_to_coarse_t = torch.tensor(fine_to_coarse, dtype=torch.long, device=device)

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        # Super Class Density: fraction of top-5 preds sharing superclass with target
        if fine_to_coarse is not None:
            _, top5_preds = output.topk(5, dim=1)  # (B, 5)
            target_sc = fine_to_coarse_t[target]  # (B,)
            pred_sc = fine_to_coarse_t[top5_preds]  # (B, 5)
            sc_match = (pred_sc == target_sc.unsqueeze(1)).float()  # (B, 5)
            sc_density = sc_match.mean(dim=1).mean().item() * 100  # percentage
            metric_logger.meters['sc_density'].update(sc_density, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if fine_to_coarse is not None:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} SC_Density {scd.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5,
                      scd=metric_logger.sc_density, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def get_acc(data_loader, model, device, keep_rate=None, tokens=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, keep_rate, tokens)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return metric_logger.acc1.global_avg
