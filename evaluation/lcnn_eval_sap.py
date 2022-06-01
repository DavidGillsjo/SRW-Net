from parsing.config import cfg
from parsing.config.paths_catalog import DatasetCatalog
import parsing.utils.metric_evaluation as me
from parsing.utils.labels import LabelMapper
import argparse
import os
import os.path as osp
import numpy as np
import json
import matplotlib.pyplot as plt
import re
from test import ModelTester
from tqdm import tqdm

def read_lcnn(res_folder, gt_folder):
    preds = sorted(glob.glob(osp.join(res_folder, '*.npz')))
    gts = sorted(glob.glob(osp.join(gt_folder, '*.npz')))
    res = []
    for p_name, gt_name in zip(preds, gts):
        img_name = gt_name.split('_0')[0] + '.png'

    return preds, gts

def adjust_lcnn(npz_folder):
    res = []
    for filename in osp.listdir(npz_folder):
        with np.load(filename) as npz:
            res.append({
                'lines_label_score': npz['score']
                'lines_valid_score': npz['score']
                'lines_label' = [1] * len(npz['score'])
                'lines': npz['lines']
            })
    return res

if __name__ == "__main__":
    argparser = argparse.ArgumentParser('Structural AP Evaluation')
    argparser.add_argument("config_file",
                        help="path to config file",
                        type=str)
    argparser.add_argument('result_folder', type=str)
    argparser.add_argument('gt_folder', type=str)
    argparser.add_argument('output_dir', type=str)
    argparser.add_argument('-l', '--write-latex', action='store_true', help='Make LaTeX tables')

    args = argparser.parse_args()

    res_filenames, gt_filenames = read_lcnn(args.result_folder, args.gt_folder)

    res = adjust_lcnn(args.npz_folder)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    dataset_name = cfg.DATASETS.TEST

    ann_file = DatasetCatalog.get(dataset_name)['args']['ann_file']

    with open(ann_file,'r') as _ann:
        annotations_list = json.load(_ann)

    tester = ModelTester(cfg,
                         output_dir = output_dir,
                         validation = 'val' in dataset_name,
                         nbr_plots = 0,
                         write_latex = args.write_latex,
                         write_tb = not args.skip_tb)


    annotations_dict = {
        ann['filename']: ann for ann in annotations_list
    }

    for jfile in tqdm(json_files):
        match = re.search(r'epoch(\d+)\.json', jfile)
        epoch = int(match.group(1)) if match else 0

        with open(jfile, 'r') as f:
            res = json.load(f)

        if args.hawp:
            adjust_hawp(res)

        tester.run_nms(res)
        sAP_dict, jAP_dict = tester.eval_sap(res, annotations_dict, epoch)
        # tester.eval_lsun_kp(res, annotations_dict, epoch)
