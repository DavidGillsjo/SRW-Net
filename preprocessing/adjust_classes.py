import os
import os.path as osp
import cv2
import matplotlib
# matplotlib.use('Cairo')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import numpy as np
import argparse
import yaml
import multiprocessing as mp
import time
import json

ORIGIN_LINE_CLASSES = (
    frozenset(['invalid']), #Never output, just for training purposes.
    frozenset(['room']),
    frozenset(['window']),
    frozenset(['door']),
    frozenset(['room', 'window']),
    frozenset(['room', 'door']),
    frozenset(['window', 'door'])
)

NEW_SINGLE_LINE_CLASSES = (
    'invalid', #Never output, just for training purposes.
    'room',
    'window',
    'door'
)

MAPPING = {
    frozenset(['invalid']): 'invalid',
    frozenset(['room']) : 'room',
    frozenset(['window']): 'window',
    frozenset(['door']): 'door',
    frozenset(['room', 'window']): 'window',
    frozenset(['room', 'door']): 'door',
    frozenset(['window', 'door']): 'door'
}

IDX_MAPPING = [NEW_SINGLE_LINE_CLASSES.index(MAPPING[l]) for l in ORIGIN_LINE_CLASSES]


def adjust_json(json_path, out_dir):
    with open(json_path, 'r') as f:
        ann = json.load(f)

    for a in ann:
        a['edges_semantic'] = [IDX_MAPPING[idx] for idx in a['edges_semantic']]
        # a['junc_occluded'] = [1]*len(a['junc_occluded'])

    out_json_path = osp.join(
        out_dir,
        osp.basename(json_path)
    )
    with open(out_json_path, 'w') as f:
        json.dump(ann, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adjust labels according to mapping')
    parser.add_argument('json_dir', type=str, help='Path to jsons')
    parser.add_argument('--out', type=str, default = None, help='Path to output dir, will be json_dir_adjusted as default')
    parser.add_argument('-j', '--nbr-workers', type=int, default = 1, help='Number of processes to split work on')

    args = parser.parse_args()

    out = args.out if args.out else osp.realpath(args.json_dir) + '_adjusted'
    os.makedirs(out)

    result = []
    with mp.Pool(processes=args.nbr_workers) as pool:
        for json_path in os.listdir(args.json_dir):
            f_args = (osp.join(args.json_dir, json_path), out)
            r = pool.apply_async(adjust_json, f_args)
            result.append(r)
        for r in tqdm(result):
            r.get()
