#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Written by yangmin09

"""Tools for training and testing a model."""

import os

import numpy as np
import core.benchmark as benchmark
import core.builders as builders
import core.checkpoint as checkpoint
import core.config as config
import core.distributed as dist
import core.logging as logging
import core.meters as meters
import core.net as net
import core.optimizer as optim
import datasets.supcon_loader as supcon_loader
import torch
from core.config import cfg

from model.supcon_delg_model import Delg

logger = logging.get_logger(__name__)



def setup_env():
    """Sets up environment for training or testing."""
    if dist.is_master_proc():
        # Ensure that the output dir exists
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        # Save the config
        config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg))
    logger.info(logging.dump_log_data(cfg, "cfg"))
    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK


def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    model = Delg()

    logger.info("Model:\n{}".format(model))
    # Log model complexity
    #logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    # Transfer the model to the current GPU device
    err_str = "Cannot use more GPU devices than available"
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device, find_unused_parameters=True
        )
        # Set complexity function to be module's complexity function
        #model.complexity = model.module.complexity
    return model

from ipdb import set_trace
def train_epoch(train_loader, model, con_loss_fun, loss_fun, optimizer, train_meter, cur_epoch):
    """Performs one epoch of training."""
    # Shuffle the data
    # supcon_loader.shuffle(train_loader, cur_epoch)
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    # train_meter.iter_tic()
    for cur_iter, (images1, images2, labels) in enumerate(train_loader):
        # Transfer the data to the current GPU device
        batch_size = cfg.TRAIN.BATCH_SIZE
        images = torch.cat([images1, images2], dim=0)
        images, labels = images.cuda(), labels.cuda(non_blocking=True)
        
        # Perform the forward pass
        global_features, global_logits, local_features, local_logits, att_scores = model(images, labels)
        
        # Perform the backward pass
        optimizer.zero_grad()

        # Compute the contrastive loss
        f1, f2 = torch.split(global_features, [batch_size, batch_size], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        con_loss = con_loss_fun(features, labels)
        print(con_loss)
        # set_trace()

        # Compute the classification loss
        # Update global logits and local logits
        gl1, gl2 = torch.split(global_logits, [batch_size, batch_size], dim=0)
        global_logits = (gl1 + gl2) / 2
        ll1, ll2 = torch.split(local_logits, [batch_size, batch_size], dim=0)
        local_logits = (ll1 + ll2) / 2
        desc_loss = loss_fun(global_logits, labels)
        att_loss = loss_fun(local_logits, labels)
        
        # Freeze localmodel before
        net.freeze_weights(model, freeze=['localmodel', 'att_cls'])
        total_global_loss = con_loss * cfg.TRAIN.CON_WEIGHT + desc_loss
        total_global_loss.backward()

        # Freeze globalmodel and unfreeze localmodel
        net.freeze_weights(model, freeze=['globalmodel', 'desc_cls']) 
        net.unfreeze_weights(model, freeze=['localmodel', 'att_cls'])
        att_loss.backward()

        net.unfreeze_weights(model, freeze=['globalmodel', 'desc_cls'])

        # update params
        optimizer.step()

        # Compute the errors
        desc_top1_err, desc_top5_err = meters.topk_errors(global_logits, labels, [1, 5])
        desc_loss, desc_top1_err, desc_top5_err = dist.scaled_all_reduce([desc_loss, desc_top1_err, desc_top5_err])
        desc_loss, desc_top1_err, desc_top5_err = desc_loss.item(), desc_top1_err.item(), desc_top5_err.item()

        att_top1_err, att_top5_err = meters.topk_errors(local_logits, labels, [1, 5])
        att_loss, att_top1_err, att_top5_err = dist.scaled_all_reduce([att_loss, att_top1_err, att_top5_err])
        att_loss, att_top1_err, att_top5_err = att_loss.item(), att_top1_err.item(), att_top5_err.item()
        
        train_meter.iter_toc()
        # Update and log stats
        mb_size = labels.size(0) * cfg.NUM_GPUS
        train_meter.update_stats(desc_top1_err, desc_top5_err, att_top1_err, att_top5_err, desc_loss, att_loss, lr, mb_size)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()




def train_model():
    """Trains the model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model, loss_fun, and optimizer
    model = setup_model()
    con_loss_fun = builders.build_supcon_loss_fun()
    loss_fun = builders.build_loss_fun().cuda()
    optimizer = optim.construct_optimizer(model)
    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and checkpoint.has_checkpoint():
        last_checkpoint = checkpoint.get_last_checkpoint()
        checkpoint_epoch = checkpoint.load_checkpoint(last_checkpoint, model, optimizer)
        logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
        start_epoch = int(checkpoint_epoch['epoch']) + 1
    elif cfg.TRAIN.WEIGHTS:
        checkpoint.load_checkpoint(cfg.TRAIN.WEIGHTS, model)
        logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
    # Create data loaders and meters
    train_loader = supcon_loader.construct_train_loader()
    # test_loader = loader.construct_test_loader()
    train_meter = meters.TrainMeter(len(train_loader))
    # test_meter = meters.TestMeter(len(test_loader))
    # Compute model and loader timings
    #if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
        #benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        train_epoch(train_loader, model, con_loss_fun, loss_fun, optimizer, train_meter, cur_epoch)
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            net.compute_precise_bn_stats(model, train_loader)
        # Save a checkpoint
        if (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            checkpoint_file = checkpoint.save_checkpoint(model, optimizer, cur_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        # Evaluate the model
        # next_epoch = cur_epoch + 1
        # if next_epoch % cfg.TRAIN.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
        #     test_epoch(test_loader, model, test_meter, cur_epoch)


