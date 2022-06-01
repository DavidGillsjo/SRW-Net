import json
import scipy.io as sio
import os
import os.path as osp
import cv2
import matplotlib
# matplotlib.use('Cairo')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import shutil
import multiprocessing as mp
import time
from datetime import timedelta
from tqdm import tqdm
import logging
import re
import errno
import random
from structured3D2wireframe import generate_negative_edges, autolabel_bar
import skimage.draw as skd
from math import floor
import yaml

class Layout:
    frontal = 1
    left = 2
    right = 3
    floor = 4
    ceiling = 5

SEGMENTATION_CLASSES = [
    'background',
    'wall',
    'floor',
    'ceiling'
]
CLASS2IDX = {c: idx for idx, c in enumerate(SEGMENTATION_CLASSES)}

class RoomTypes:
    def __init__(self, mat_file):
        type_struct = sio.loadmat(mat_file, squeeze_me=True)['type']
        self.edge_map = {t['typeid']: (t['lines']-1).reshape(-1,2) for t in type_struct}

    def type2edges(self, type):
        return self.edge_map[type]

def classify_lines(seg_img, junctions, edges):
    labels = []
    clipped_junctions = np.clip(junctions, [0,0], np.array(seg_img.shape[::-1]) - 0.51)
    for edge in edges:
        jpos_int = clipped_junctions[edge].astype(np.int)
        rr, cc, value = skd.line_aa(*jpos_int[0,::-1], *jpos_int[1,::-1])
        counts = np.bincount(seg_img[rr,cc], minlength=6)
        total = np.sum(counts)

        if counts[Layout.floor]/total > 0.2:
            labels.append(CLASS2IDX['floor'])
        elif counts[Layout.ceiling]/total > 0.2:
            labels.append(CLASS2IDX['ceiling'])
        else:
            labels.append(CLASS2IDX['wall'])

    return labels

def lsun_mat2ann(record, data_dir, out_image_dir, room_types, plot_dir = None):
    logger = logging.getLogger('LSUN2wireframe')
    annotations = []
    for r in record:
        ann = {}
        # Link image in new dir
        ann['filename'] = r['image'] + '.jpg'
        ann['seg_filename'] = r['image'] + '.mat'
        ann['height'], ann['width'] = r['resolution'].tolist()
        for fname in (ann['filename'], ann['seg_filename']):
            try:
                #Remove if existing
                os.remove(osp.abspath(osp.join(out_image_dir, fname)))
            except IOError:
                pass
            os.symlink(
                osp.relpath(osp.join(data_dir, fname), start = out_image_dir),
                osp.join(out_image_dir, fname))
        # Get junctions
        ann['junctions'] = r['point'].tolist()
        ann['junctions_semantic'] = [1]*len(ann['junctions'])

        # Get Edges
        edges = room_types.type2edges(r['type'])
        ann['edges_positive'] = edges.tolist()

        sem_img = sio.loadmat(osp.join(data_dir, r['image'] + '.mat'))['layout']
        ann['edges_semantic'] = classify_lines(sem_img, r['point'], edges)

        img = cv2.imread(osp.join(data_dir, ann['filename'] ))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ann['edges_negative'] = generate_negative_edges(img, r['point'], edges)

        annotations.append(ann)

        if plot_dir:
            plt.figure()
            plt.imshow(img)
            sem_colors = ('tab:orange', 'tab:purple', 'tab:green', 'tab:orange')
            junc_np = np.array(ann['junctions']).T
            junc_sem_np = np.array(ann['junctions_semantic'], dtype=np.int)
            for edge, sem in zip(ann['edges_positive'], ann['edges_semantic']):
                plt.plot((junc_np[0,edge[0]], junc_np[0, edge[1]]),
                         (junc_np[1,edge[0]], junc_np[1, edge[1]]), color=sem_colors[sem])
            plt.plot(*junc_np[:,junc_sem_np==0], '.', color='tab:blue')
            plt.plot(*junc_np[:,junc_sem_np==1], 'r.', color='tab:red')


            plt.savefig(osp.join(plot_dir, ann['filename']))
            plt.close()


    return annotations

if __name__ == '__main__':
    script_path = osp.dirname(osp.realpath(__file__))
    parser = argparse.ArgumentParser(description='Generate wireframe format from LSUN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_dir', type=str, help='Path to LSUN')
    parser.add_argument('out_dir', type=str, help='Path for storing conversion')
    parser.add_argument('-j', '--nbr-workers', type=int, default = 1, help='Number of processes to split work on')
    parser.add_argument('-s', '--nbr-scenes', type=int, default = None, help='Number of scenes to process')
    parser.add_argument('-l', '--logfile', type=str, default = None, help='logfile path if wanted')
    parser.add_argument('--room-types', type=str, help='Matlab Room type struct',
                        default = osp.abspath(osp.join(script_path, 'LSUN_room_types.mat')))
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite out_dir if existing')
    parser.add_argument('--halt', action='store_true', help='Halt on error')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot images with GT lines')

    args = parser.parse_args()

    # create logger
    logger = logging.getLogger('LSUN2wireframe')
    logger.propagate = True
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if args.logfile:
        fh = logging.FileHandler(args.logfile, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if osp.exists(args.out_dir) and not args.overwrite:
        print("Output directory {} already exists, specify -o flag if overwrite is permitted".format(args.out_dir))
        sys.exit()

    room_types = RoomTypes(args.room_types)

    out_image_dir = osp.join(args.out_dir, 'images')
    os.makedirs(out_image_dir, exist_ok = True)

    if args.plot:
        plot_dir = '/host_home/plots/LSUN'
        os.makedirs(plot_dir, exist_ok=True)
    else:
        plot_dir = None

    segmentation_dir = osp.join(args.data_dir, 'surface_relabel')
    dset2mat = {'val': 'validation', 'train': 'training'}
    img_folder = osp.join(args.data_dir, 'surface_relabel')


    for dset, matdset in dset2mat.items():
        kp_mat = sio.loadmat(osp.join(args.data_dir, '{}.mat'.format(matdset)), squeeze_me=True)
        kp_mat = kp_mat[matdset]
        if args.nbr_scenes:
            kp_mat = kp_mat[:args.nbr_scenes]
        full_img_folder = osp.join(img_folder, dset)
        result = []
        ann = []
        chunk_size = min(5, floor(len(kp_mat)/args.nbr_workers))
        chunk_size = max(chunk_size, 1)
        with mp.Pool(processes=args.nbr_workers) as pool:
            for idx in range(0, len(kp_mat), chunk_size):
                f_args = (kp_mat[idx:idx+chunk_size], full_img_folder, out_image_dir, room_types)
                f_kwargs = dict(plot_dir=plot_dir)
                r = pool.apply_async(lsun_mat2ann, f_args, f_kwargs)
                result.append(r)

            #Wait for results, waits for processes to finish and raises errors
            for r in tqdm(result):
                try:
                    ann += r.get()
                except KeyboardInterrupt:
                    sys.exit()
                except:
                    logger.exception('Got exception')
                    raise

        with open(osp.join(args.out_dir, '{}.json'.format(dset)), 'w') as f:
            json.dump(ann, f)

        nbr_images = len(ann)
        r = {}
        r['nbr_junctions'] = np.array([len(a['junctions']) for a in ann])
        r['nbr_visible_junctions'] = np.array([(np.array(a['junctions_semantic'], dtype=np.int)).sum() for a in ann])
        r['nbr_edges_pos'] = np.array([len(a['edges_positive']) for a in ann])
        r['nbr_edges_neg'] = np.array([len(a['edges_negative']) for a in ann])

        fig, ax = plt.subplots(2,3)
        for i,title in enumerate(r):
            ax1 = ax.flat[i]
            ax1.hist(r[title], bins=15)
            ax1.set_title(title)
            ax1.set_ylabel('Nbr images (total: {})'.format(nbr_images))
            ax1.set_xlabel('{} / image'.format(title))

        line_c = np.concatenate([a['edges_semantic'] for a in ann])
        ax1 = ax.flat[-1]
        hist, bin_edges = np.histogram(line_c, bins=range(len(SEGMENTATION_CLASSES)+1))
        bar_plot = ax1.bar(bin_edges[:-1], hist, align='center')
        autolabel_bar(ax1, bar_plot, hist)
        ax1.set_title('line labels')
        ax1.set_ylabel('Nbr Lines')
        ax1.set_xticks(range(len(SEGMENTATION_CLASSES)))
        ax1.set_xticklabels(['-'.join(c) for c in SEGMENTATION_CLASSES], rotation=45, ha='right')
        ax1.set_ylim([0, np.max(hist)*1.5])
        plt.tight_layout()
        plt.savefig(osp.join(args.out_dir, 'stats_{}.svg'.format(dset)))
        plt.close()

        total_instances = np.sum(hist)
        nbr_fake_invalid = total_instances
        fake_total = total_instances + nbr_fake_invalid
        fake_hist = np.copy(hist)
        fake_hist[0] = nbr_fake_invalid
        weights = (1.0/fake_hist)*fake_total/len(SEGMENTATION_CLASSES)
        weights[fake_hist == 0] = 1
        bias = fake_hist/fake_total
        # YAML stats
        stats = {
            'class_names': SEGMENTATION_CLASSES,
            'nbr_line_classes': [int(N) for N in hist],
            'weight_line_classes': [float(N) for N in weights],
            'bias_line_classes': [float(N) for N in bias],
        }
        print(stats)
        for i in range(len(SEGMENTATION_CLASSES)):
            print('{} - N:{}, W:{:.02g}, B:{:.02g}'.format(
                stats['class_names'][i],
                stats['nbr_line_classes'][i],
                stats['weight_line_classes'][i],
                stats['bias_line_classes'][i],
            ))
        with open(osp.join(args.out_dir, 'stats_{}.yaml'.format(dset)), 'w') as f:
            yaml.safe_dump(stats, f, default_flow_style=None)
