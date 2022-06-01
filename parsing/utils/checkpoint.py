# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch

from parsing.utils.model_serialization import load_state_dict
from parsing.utils.c2_model_loading import load_c2_format
from parsing.utils.imports import import_file
from parsing.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
        wandb_id=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        self.wandb_id = wandb_id
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        if self.wandb_id is not None:
            data["wandb_id"] = self.wandb_id
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, f_gnn=None, use_latest=True, transfer = False):
        if self.has_checkpoint() and use_latest:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        if f_gnn:
            gnn_checkpoint = self._load_file(f_gnn)
            checkpoint = self._merge_models(checkpoint, gnn_checkpoint)
        self._load_model(checkpoint, transfer=transfer)
        if "optimizer" in checkpoint and self.optimizer and not transfer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler and not transfer:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    # Assumes we convert WireframeDetector and WireframeGNNClassifier to WireframeDetector with WireframeGNNHead
    def _merge_models(self, checkpoint, gnn_checkpoint):
        new_model = {}
        model = checkpoint['model']
        for k,v in model.items():
            if not k.startswith('gnn'):
                new_model[k] = v
        gnn_model = gnn_checkpoint['model']
        for k,v in gnn_model.items():
            if k.startswith('gnn_head'):
                new_model[k] = v
            elif k.startswith('gnn.'):
                new_model['gnn_head.line_gnn.' + k[4:]] = v
            else:
                new_model['gnn_head.' + k] = v
        # print('model', model.keys())
        print('gnn_model', {k.split('.')[0] for k in gnn_model.keys()})
        print('model', {k.split('.')[0] for k in model.keys()})
        print('new_model', {k.split('.')[0] for k in new_model.keys()})
        checkpoint['model'] = new_model
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint, transfer=True):
        if transfer:
            state_dict = {
                k:v for (k,v) in checkpoint.pop("model").items()
                if ('head' not in k) and ('backbone.score' not in k) and ('fc2' not in k)
                }
        else:
            state_dict = checkpoint.pop("model")
        load_state_dict(self.model, state_dict, strict=not transfer)


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded
