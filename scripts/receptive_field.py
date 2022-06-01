import torch
import parsing
from parsing.config import cfg
from parsing.utils.comm import to_device
from parsing.dataset import build_test_dataset
from parsing.detector import WireframeDetector
from parsing.utils.logger import setup_logger
from parsing.utils.metric_logger import MetricLogger
from parsing.utils.miscellaneous import save_config
from parsing.utils.checkpoint import DetectronCheckpointer
import os
import os.path as osp
import time
import datetime
import argparse
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
parser = argparse.ArgumentParser(description='Visualize receptive field for backbone')

parser.add_argument("--config-file",
                    metavar="FILE",
                    help="path to config file",
                    type=str,
                    required=True,
                    )
parser.add_argument("opts",
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER
                    )
args = parser.parse_args()


def test(cfg):
    logger = logging.getLogger("hawp.rf")
    device = cfg.MODEL.DEVICE
    model = WireframeDetector(cfg)
    model = model.to(device)

    checkpointer = DetectronCheckpointer(cfg,
                                         model,
                                         save_dir=cfg.OUTPUT_DIR,
                                         save_to_disk=True,
                                         logger=logger)
    if getattr(cfg, 'CHECKPOINT', None):
        checkpointer.load(cfg.CHECKPOINT, use_latest=False)

    image = torch.zeros((1, 3, cfg.DATASETS.IMAGE.HEIGHT, cfg.DATASETS.IMAGE.WIDTH), dtype=torch.float32)
    image[0,:,cfg.DATASETS.IMAGE.HEIGHT//2, cfg.DATASETS.IMAGE.WIDTH//2] = 1.0
    image_np = image.numpy()

    with torch.no_grad():
        outputs, features = model.backbone(image.to(device))
    step_response = features.cpu().numpy()

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image[0,0])
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(step_response[0].mean(axis=0))
    plt.colorbar()
    plt.savefig('receptive_field.png')
    plt.close()


if __name__ == "__main__":

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger('hawp', output_dir)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))


    test(cfg)

    ### Training



    # import pdb; pdb.set_trace()
