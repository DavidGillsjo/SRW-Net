# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys
import wandb
import os.path as osp

def setup_logger(name, save_dir, out_file='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, out_file))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

# Set your key and project details for for wandb logging
#os.environ["WANDB_API_KEY"] = ""
project = "semantic-room-wireframe"
entity = "room-wireframe"

def wandb_init(cfg, checkpointer, resume=False, timestamp = ''):
    if resume:
        wandb_run = _wandb_from_checkpoint(checkpointer, cfg)
    else:
        wandb_run = _wandb_from_config(cfg, timestamp)
    checkpointer.wandb_id = wandb_run.id
    return wandb_run

def _wandb_from_config(cfg, timestamp):
    kwargs = dict(
        name = f'{cfg.EXPERIMENT.NAME}-{timestamp}',
        group = cfg.EXPERIMENT.GROUP,
        notes = cfg.EXPERIMENT.NOTES,
        dir = osp.join(cfg.OUTPUT_DIR),
        project = project,
        entity = entity,
        resume = 'never'
    )
    return wandb.init(**kwargs)

def _wandb_from_checkpoint(checkpoint, cfg):
    assert checkpoint.wandb_id
    kwargs = dict(
        id = checkpoint.wandb_id,
        dir = osp.join(cfg.OUTPUT_DIR),
        project = project,
        entity = entity,
        resume = 'must'
    )
    return wandb.init(**kwargs)
