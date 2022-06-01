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
from glob import glob

class AnnData:
    def __init__(self, nbr_junctions = []):
        self.nbr_junctions = nbr_junctions

    def __add__(self,other):
        return AnnData(self.nbr_junctions + other.nbr_junctions)

class AnnStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.all_nbr_junctions = []

    def add(self, ann):
        for a in ann:
            self.all_nbr_junctions.append(len(a['junctions']))

    def add_ann_from_path(self, ann_path):
        with open(ann_path) as f:
            ann = yaml.safe_load(f)
        self.add(ann)

    def add_files(self, ann_paths, nbr_workers = 1):
        for a_path in tqdm(ann_paths):
            self.add_ann_from_path(a_path)

    # def _compute_stats(self, ann):


    def write_yaml(self, yaml_path, plot_path):
        plt.figure()
        result = plt.boxplot(self.all_nbr_junctions)
        plt.savefig(plot_path)
        plt.close()
        print(result)
        print(result['means'])
        yaml_dict = {
            'JUNCTION_MEAN': float(np.mean(self.all_nbr_junctions)),
            'JUNCTION_MEDIAN': float(np.median(self.all_nbr_junctions)),
        }
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(yaml_dict, f)
        return yaml_dict



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate stats for annotations')
    parser.add_argument('ann_files', type=str, nargs='+', help='Annotation files')
    parser.add_argument('--out', type=str, default = 'stats.yaml', help='Path to output yaml')
    parser.add_argument('-j', '--nbr-workers', type=int, default = 1, help='Number of processes to split work on')

    args = parser.parse_args()

    t = time.time()
    stats = AnnStats()
    stats.add_files(args.ann_files, nbr_workers = args.nbr_workers)
    result = stats.write_yaml(args.out, args.out + '.svg')
    print('Took {}s'.format(time.time()-t))
    print(result)
