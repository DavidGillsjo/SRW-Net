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

def adjust_hawp(res):
    for output in res:
        output['lines_label_score'] = output['lines_score']
        output['lines_valid_score'] = output['lines_score']
        output['lines_label'] = [1] * len(output['lines_score'])
        output['juncs_label_score'] = output['juncs_score']
        output['juncs_valid_score'] = output['juncs_score']
        output['juncs_label'] = [1] * len(output['juncs_score'])

if __name__ == "__main__":
    argparser = argparse.ArgumentParser('Structural AP Evaluation')
    argparser.add_argument("config_file",
                        help="path to config file",
                        type=str,
                        )
    argparser.add_argument('path', type=str)
    argparser.add_argument('-o', '--output-dir', type=str, default=None, help='Defaults to same as result folder')
    argparser.add_argument('-d', '--dataset', type=str, default=None, help='Force use of dataset')
    argparser.add_argument('-t', '--skip-tb', action='store_true', help='Skip logging to tensorboard')
    argparser.add_argument('-l', '--write-latex', action='store_true', help='Make LaTeX tables')
    argparser.add_argument('--hawp', action='store_true', help='HAWP input')
    argparser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )

    args = argparser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    result_path = args.path

    if result_path.endswith('.json'):
        batch_eval = False
        output_dir = osp.join(osp.dirname(result_path), 'eval')
        json_files = [result_path]
    else:
        batch_eval = True
        output_dir = osp.join(result_path, 'eval')
        json_files = []
        for root, dirnames, filenames in os.walk(result_path):
            json_files += [osp.join(root, f) for f in filenames if f.endswith('.json')]

        json_files.sort()

    if args.output_dir:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    assert len(json_files) > 0, 'The result file has to end with .json'

    if args.dataset:
        dataset_name = args.dataset
    else:
        dataset_name = re.sub(r'(_epoch\d+)*\.json', '', osp.basename(json_files[0]))

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
