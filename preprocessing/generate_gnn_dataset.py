import torch
import parsing
from parsing.config import cfg
from parsing.utils.comm import to_device
from parsing.dataset import build_generate_dataset
from parsing.detector import WireframeDetector
from parsing.utils.logger import setup_logger
from parsing.utils.metric_logger import MetricLogger
from parsing.utils.miscellaneous import save_config
from parsing.utils.checkpoint import DetectronCheckpointer
from parsing.utils.labels import LabelMapper
from torch.utils.tensorboard import SummaryWriter
import parsing.utils.metric_evaluation as me
import os
import os.path as osp
from skimage import io
import time
import datetime
import argparse
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from glob import glob
import numpy as np
from tabulate import tabulate
from parsing.dataset.transforms import Compose, ResizeImage, ToTensor, Normalize



# Handles a single process to test and evaluate
# TODO: Evaluate model in hogwild style, then return to caller and process result and plots in background.
class DataGenerator:
    def __init__(self, cfg, output_dir = None, use_gt = False):
        self.cfg = cfg
        self.output_dir = output_dir
        self.datasets = build_generate_dataset(cfg)
        self.logger = logging.getLogger("hawp.generate_gnn")

        self.img_transform =  Compose(
            [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                         cfg.DATASETS.IMAGE.WIDTH),
             ToTensor(),
             Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                       cfg.DATASETS.IMAGE.PIXEL_STD,
                       cfg.DATASETS.IMAGE.TO_255)
            ]
        )
        self.lm = LabelMapper(cfg.MODEL.LINE_LABELS, cfg.MODEL.JUNCTION_LABELS, disable=cfg.DATASETS.DISABLE_CLASSES)
        if use_gt:
            self.gen_f = self._make_gt_data
        else:
            self.gen_f = self._make_pred_data


    def _load_model_from_cfg(self):
        model = WireframeDetector(self.cfg)
        device = self.cfg.MODEL.DEVICE
        model = model.to(device)
        checkpointer = DetectronCheckpointer(self.cfg,
                                             model,
                                             save_dir=self.cfg.OUTPUT_DIR,
                                             save_to_disk=True,
                                             logger=self.logger)


        if getattr(cfg, 'CHECKPOINT', None):
            checkpointer.load(f=cfg.CHECKPOINT, use_latest=False)
        else:
            checkpointer.load()

        return model

    def _make_pred_data(self, model, images, annotations, outpath_npy, device = 'cuda'):
        ann = annotations[0]
        with torch.no_grad():
            output, extra_info = model(images.to(device), annotations, output_features=True)

        if extra_info['gnn_line_features'] is not None:
            assert output['lines_pred'].shape[0] == extra_info['gnn_line_features'].shape[0]
            assert output['juncs_pred'].shape[0] == extra_info['gnn_junction_features'].shape[1]
            np.savez(outpath_npy,
                     line_coordinates = output['lines_pred'].to('cpu').numpy(),
                     line_features = extra_info['gnn_line_features'].to('cpu').numpy(),
                     junction_features = extra_info['gnn_junction_features'].to('cpu').numpy(),
                     junction_idx = output['line2junc_idx'].to('cpu').numpy(),
                     junction_coordinates = output['juncs_pred'].to('cpu').numpy()
                     )




    def _make_gt_data(self, model, images, annotations, outpath_npy, device = 'cuda'):
        ann = annotations[0]
        with torch.no_grad():
            images = images.to(device)
            outputs, features = model.backbone(images)
            loi_features = model.fc1(features)
            sx = 128.0/float(ann['width'])
            sy = 128.0/float(ann['height'])
            junctions = ann['junctions']

            edges = torch.cat((ann['edges_positive'],ann['edges_negative']))
            lines = torch.cat((junctions[edges[:,0]], junctions[edges[:,1]]),dim=-1).to(device)

            # cmax = torch.tensor(127,dtype=torch.long)
            # junctions_rescaled = junctions.clone()
            # junctions_rescaled[:,0] *= 128/float(ann['width'])
            # junctions_rescaled[:,1] *= 128/float(ann['height'])
            # xint = torch.minimum(junctions_rescaled[:,0].long(),cmax)
            # yint = torch.minimum(junctions_rescaled[:,1].long(),cmax)
            junctions_128 = junctions.clone()
            junctions_128[:,0] = torch.clamp(junctions_128[:,0]*sx, 0, 128-1e-4)
            junctions_128[:,1] = torch.clamp(junctions_128[:,1]*sy, 0, 128-1e-4)
            lines_128 = torch.cat((junctions_128[edges[:,0]], junctions_128[edges[:,1]]),dim=-1).to(device)
            xint = junctions_128[:,0].long()
            yint = junctions_128[:,1].long()
            assert torch.all(xint < 128)
            assert torch.all(xint >= 0)
            assert torch.all(yint < 128)
            assert torch.all(yint >= 0)
            flat_index = yint*128 + xint
            logits, pooled_line_features, junction_features = model.pooling(loi_features[0],lines_128, flat_index, edges)
            np.savez(outpath_npy,
                     line_coordinates = lines.to('cpu').numpy(),
                     line_features = pooled_line_features.to('cpu').numpy(),
                     junction_features = junction_features.to('cpu').numpy(),
                     junction_idx = edges.to('cpu').numpy(),
                     junction_coordinates = junctions.to('cpu').numpy()
                     )

    def generate_data(self, model = None):
        if model is None:
            model = self._load_model_from_cfg()
        model.eval()
        device = self.cfg.MODEL.DEVICE

        for datatuple in self.datasets:
            try:
                name, dataset = datatuple
            except ValueError:
                name = 'Training'
                dataset = datatuple

            self.logger.info('Generating from {} dataset'.format(name))
            skipped = 0

            for i, (images, annotations) in enumerate(tqdm(dataset)):
                assert len(annotations) == 1
                ann = annotations[0]
                outpath_npy = osp.join(self.output_dir,'{}.npz'.format(ann['filename']))
                if osp.exists(outpath_npy):
                    skipped += 1
                    continue
                self.gen_f(model, images, annotations,outpath_npy, device = device)
            self.logger.info('Skipped {} files from {} dataset'.format(skipped, name))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate GNN data')

    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        required=True,
                        )

    parser.add_argument("--output-dir",
                        default='npz',
                        type=str,
                        help='Output dir')


    parser.add_argument("--use-gt",
                        action='store_true',
                        help='Use GT lines, positive and negative')

    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('hawp', args.output_dir)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))


    generator = DataGenerator(cfg,
                              output_dir = args.output_dir,
                              use_gt = args.use_gt)
    generator.generate_data()
