import json
import yaml
import scipy.io as sio
import os
import os.path as osp
import cv2
import skimage.draw
import scipy.ndimage
from itertools import combinations, combinations_with_replacement
import matplotlib
# matplotlib.use('Cairo')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse
import csv
import sys
import shutil
import shapely.geometry as sg
import shapely
import open3d
from scipy.spatial.distance import cdist
from copy import deepcopy
import cProfile, pstats
import multiprocessing as mp
import time
from datetime import timedelta
from tqdm import tqdm
import logging
import re
import errno
import random
from parsing.utils.visualization import ImagePlotter, set_fonts
from collections import deque
from descartes import PolygonPatch
import seaborn as sns
from structured3D2wireframe import ALL_SINGLE_LINE_CLASSES, plane2label,autolabel_bar
from structured3D_geometry import DIV_EPS
from tabulate import tabulate

#Plot settings
sns.set_theme()

set_fonts()

STAT_HIST_BINS = {
    'Max Distance':range(0,80, 5),
    'Median Distance':range(0,40, 5),
    'Min Distance':range(0,10),
    '# Lines':range(0,10),
    '# Junctions':range(0,10)
}

STAT_HIST_XTICKS = {s:STAT_HIST_BINS[s]  for s in ['# Lines', '# Junctions']}
STAT_HIST_XTICKS['Max Distance'] = range(0,80,20)
STAT_HIST_XTICKS['Median Distance'] = range(0,40,10)
STAT_HIST_XTICKS['Min Distance'] = range(0,12,2)


STAT_HIST_XTICKS.update( {s:range(0,STAT_HIST_BINS[s][-1]+1,20)  for s in ['Max Distance']} )
STAT_HIST_KWARGS = {s:{'align':'left'}  for s in ['# Lines', '# Junctions']}

def init_stats():
    types = ['wall', 'ceiling', 'floor', 'all']
    stats = {}
    for t in types:
        stats[t] = {}
        for st in STAT_HIST_BINS:
            stats[t][st] = []
    return stats

def collect_stats(root, scene_dir, estimate_plane = False):
    with open(osp.join(root, scene_dir, 'annotation_3d.json')) as f:
        ann_3D = json.load(f)

    stats = init_stats()
    line_stats = {label:0 for label in ALL_SINGLE_LINE_CLASSES}

    # Prepare ann_3D with information we want later
    line_junction_m = np.array(ann_3D['lineJunctionMatrix'], dtype=np.bool)
    plane_line_m = np.array(ann_3D['planeLineMatrix'], dtype=np.bool)
    junctions = np.array([j['coordinate'] for j in ann_3D['junctions']], dtype=np.float32).T

    line_is_door_mask = np.zeros(plane_line_m.shape[1], dtype=np.bool)
    line_is_window_mask = np.zeros_like(line_is_door_mask)
    plane_semantic = {}
    for semantic in ann_3D['semantics']:
        for id in semantic['planeID']:
            p = ann_3D['planes'][id]
            plane_semantic[id] = plane2label(p['type'], semantic['type'])
            if semantic['type'] == 'door':
                line_is_door_mask |= plane_line_m[id]
            elif semantic['type'] == 'window':
                line_is_window_mask |= plane_line_m[id]

    # line_is_normal_mask = ~line_is_door_mask & ~line_is_window_mask
    for p_mask in plane_line_m.T:
        label = frozenset([plane_semantic[idx] for idx in np.flatnonzero(p_mask)])
        if not 'outwall' in label:
            line_stats[label] += 1


    for p, pl_mask in zip(ann_3D['planes'],plane_line_m):
        sub_stat = stats[p['type']]
        p_sem = plane_semantic[p['ID']]
        # if p['type']=='wall' and p_sem not in set(['door', 'window']):
        #     pl_mask &= line_is_normal_mask
        junction_mask = np.any(line_junction_m[pl_mask], axis=0)
        num_junctions  = np.sum(junction_mask)
        pj = junctions[:,junction_mask]
        if estimate_plane and num_junctions > 2:
            params = estimate_plane_params(pj)
        else:
            params = np.array(p['normal'] + [p['offset']], dtype=np.float32)
            params /= np.linalg.norm(params[:3])

        if params.shape[0] < 4: print(params)
        pj_dist = np.abs(params[:3]@pj + params[3])
        # if np.max(pj_dist) > 1:
        sub_stat['Max Distance'].append(        np.max(pj_dist) )
        sub_stat['Median Distance'].append(     np.median(pj_dist) )
        sub_stat['Min Distance'].append(        np.min(pj_dist) )
        sub_stat['# Lines'].append(       np.sum(pl_mask) )
        sub_stat['# Junctions'].append(   num_junctions)

    return stats, line_stats

def estimate_plane_params(junctions):
    # 3xN matrix of junctions
    num_junctions = junctions.shape[1]
    assert num_junctions > 2
    junctions_h = np.vstack([junctions, np.ones(num_junctions).reshape([1,-1])])

    # Normalize
    N = np.eye(4)
    j_mean = np.mean(junctions, axis=1)
    j_std = np.std(junctions-j_mean[:,None],axis=1) + DIV_EPS
    N[:3,3] = -j_mean
    N[:3] /= j_std.reshape([3,1])
    M = N@junctions_h

    # Minimize
    U,S,V = np.linalg.svd(M.T)
    params = N.T@V[-1]
    params /= np.linalg.norm(params[:3])
    return params


def update_plane_stats(agg_stats, new_stats):
    for type, type_stat in new_stats.items():
        for stat_name, data in type_stat.items():
            agg_stats[type][stat_name] += data
            agg_stats['all'][stat_name] += data

def update_line_stats(agg_stats, new_stats):
        for label, count in new_stats.items():
            agg_stats[label] += count

def plot_plane_stats(stats, out_dir):
    sns.set_style("ticks")
    for type, type_stats in stats.items():
        plt.figure()
        plt.suptitle(type)
        # for i, sname in enumerate(sorted(type_stats.keys())):
        for i, sname in enumerate(type_stats.keys()):
            data = type_stats[sname]
            plt.subplot(2,3,i+1)
            plt.hist(data, STAT_HIST_BINS[sname], **STAT_HIST_KWARGS.get(sname, {}), label='hist.', log=True)
            plt.hist(data, STAT_HIST_BINS[sname], **STAT_HIST_KWARGS.get(sname, {}), cumulative=-1, histtype='step', label='c.hist.', log=True)
            plt.xlabel(sname)
            plt.ylabel('# Planes')
            # plt.ylim([0,300])
            if sname in STAT_HIST_XTICKS:
                plt.xticks(STAT_HIST_XTICKS[sname])
        plt.legend(bbox_to_anchor=(1.8, 1.05))
        sns.despine()
        plt.tight_layout()
        plt.savefig(osp.join(out_dir, '{}.pdf'.format(type)))
        plt.close()

def plot_line_stats(stats, out_dir):
    labels = [k for k in stats.keys() if not 'invalid' in k]
    counts = [stats[l] for l in labels]
    idx = np.argsort(counts)[::-1]
    labels = [labels[i] for i in idx]
    counts = [counts[i] for i in idx]
    sns.set_style("ticks")
    plt.figure()
    ax = plt.gca()
    bar_plot = ax.bar(range(len(labels)), counts, align='center')
    autolabel_bar(ax, bar_plot, counts)
    ax.set_xlabel('Plane Labels')
    ax.set_ylabel('# Lines')
    ax.set_xticks(range(len(labels)))
    xtick_labels = []
    for c in labels:
        if len(c) == 1:
            l = next(iter(c))
            xtick_labels.append(l+'-'+l)
        else:
            xtick_labels.append('-'.join(c))
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
    ax.set_ylim([0, np.max(counts)*1.5])

    sns.despine()
    plt.tight_layout()

    plt.savefig(osp.join(out_dir, 'lines_stats.pdf'))
    plt.close()

def write_line_stats(stats, out_dir):
    labels = [k for k in stats.keys() if not 'invalid' in k]
    counts = [stats[l] for l in labels]
    idx = np.argsort(counts)[::-1]
    counts = [counts[i] for i in idx]
    labels_str = []
    for i in idx:
        l_set = labels[i]
        if len(l_set) == 1:
            l = next(iter(l_set))
            labels_str.append(l+'-'+l)
        else:
            labels_str.append('-'.join(l_set))
    labels = labels_str
    total = np.sum(counts)
    ratio = np.array(counts)/total
    t_headers = ['label', 'count', 'ratio']
    rows = []
    for l,c,r in zip(labels, counts, ratio):
        rows.append([l, c, '{:.1f}%'.format(r*100)])
    table = tabulate(rows, headers = t_headers, tablefmt="latex_raw")

    with open(osp.join(out_dir, 'line_stats.tex'),'w') as f:
        f.write(table)

if __name__ == '__main__':
    script_path = osp.dirname(osp.realpath(__file__))
    parser = argparse.ArgumentParser(description='Generate stats on Structured3D', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_dir', type=str, help='Path to Structured3D')
    parser.add_argument('out_dir', type=str, help='Path to plots')
    parser.add_argument('-j', '--nbr-workers', type=int, default = 1, help='Number of processes to split work on')
    parser.add_argument('-s', '--nbr-scenes', type=int, default = None, help='Number of scenes to process')
    parser.add_argument('-e', '--estimate-plane', action='store_true', help='Estimate plane params')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dirs = (os.listdir(args.data_dir))

    if args.nbr_scenes:
        dirs = dirs[:args.nbr_scenes]

    result = []
    agg_plane_stats = init_stats()
    agg_line_stats = {label:0 for label in ALL_SINGLE_LINE_CLASSES}
    expected_id = 1
    missing_ids = []
    with mp.Pool(processes=args.nbr_workers) as pool:
        for scene_dir in sorted(dirs):
            scene_id = int(scene_dir.split('_')[1])
            if scene_id > expected_id:
                missing_ids.append(range(expected_id, scene_id))
            expected_id = scene_id + 1

            f_args = (args.data_dir, scene_dir)
            kw_args = {'estimate_plane': args.estimate_plane}
            r = pool.apply_async(collect_stats, f_args, kw_args)
            result.append(r)
        print('Missing IDS:')
        for mi in missing_ids:
            print(mi)

        #Wait for results, waits for processes to finish and raises errors
        for i, r in enumerate(tqdm(result)):
            try:
                plane_stat, line_stat = r.get()
                update_plane_stats(agg_plane_stats, plane_stat)
                update_line_stats(agg_line_stats, line_stat)
            except KeyboardInterrupt:
                raise

    plot_plane_stats(agg_plane_stats, args.out_dir)
    plot_line_stats(agg_line_stats, args.out_dir)
    write_line_stats(agg_line_stats, args.out_dir)
