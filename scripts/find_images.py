#!/usr/bin/python3
import re
import os
from parsing.config import cfg
import os.path as osp
import subprocess as sp
import argparse
import multiprocessing as mp
from tqdm import tqdm
import logging
import random
from parsing.utils.visualization import ImagePlotter
from parsing.utils.labels import LabelMapper
from skimage import io
import json
import shutil

TMP_ZIP_NAME = 'tmp.zip'
TMP_REPAIR_NAME = 'tmp_r.zip'


def save_images(annotations, img_dir, output_dir, image_plotter, nbr_images = None, filenames = [], filename_folder=None, show_legend = False):
    raw_dir = osp.join(output_dir,'raw')
    os.makedirs(raw_dir, exist_ok=True)
    ann_dir = osp.join(output_dir,'annotated')
    os.makedirs(ann_dir, exist_ok=True)


    if filename_folder:
        filenames = set([fn for fn in os.listdir(filename_folder) if fn.endswith('png')])
        annotations = [ann for ann in annotations if ann['filename'] in filenames]
    elif filenames:
        filenames = set(filenames)
        annotations = [ann for ann in annotations if ann['filename'] in filenames]
    elif nbr_images:
        annotations = random.sample(annotations, nbr_images)

    for ann in tqdm(annotations):
        shutil.copy(osp.join(img_dir, ann['filename']),
                    osp.join(raw_dir, ann['filename']))
        image = io.imread(osp.join(img_dir,ann['filename']))[:,:,:3]
        image_plotter.plot_gt_image(image, ann, ann_dir, desc='GT', show_legend = show_legend, ext='.pdf')




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Save images from annotation file')
    parser.add_argument('config_file', type=str, help='Path to config for label names.')
    parser.add_argument('annotation_file', type=str, help='Path to annotation.')
    parser.add_argument('output_dir', type=str, help='Path save images.')
    parser.add_argument('--nbr-ann', type=int, help='Number of annotation to save. Default is all', default=None)
    parser.add_argument('--nbr-workers', type=int, help='Number of workers . Default: %(default)s', default=1)
    parser.add_argument('-l', '--show-legend', action='store_true', help='Show legend on plots')
    parser.add_argument('--filenames', nargs='*', help='List of files to look for, overrides nbr-ann.')
    parser.add_argument('--filename-folder', type=str, help='Directoty to with images to find corresponding annotation for', default=None)

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    lm = LabelMapper(cfg.MODEL.LINE_LABELS, cfg.MODEL.JUNCTION_LABELS, disable=cfg.DATASETS.DISABLE_CLASSES)
    image_plotter = ImagePlotter(lm.get_line_labels(), lm.get_junction_labels())

    os.makedirs(args.output_dir, exist_ok=True)
    root_dir = osp.join(osp.dirname(args.annotation_file), 'images')
    with open(args.annotation_file, 'r') as f:
        annotations = json.load(f)

    save_images(annotations, root_dir, args.output_dir, image_plotter, args.nbr_ann, args.filenames, args.filename_folder, args.show_legend)
