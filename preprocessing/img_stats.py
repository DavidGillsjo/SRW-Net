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

class ImageStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.all_mean = []
        self.all_std = []
        self.Ni = 0 #Nbr Images

    def add(self, image):
        self.all_mean.append(image.mean(axis=(0,1)))
        self.all_std.append(image.std(axis=(0,1)))
        self.Ni += 1

    def add_img_from_path(self, image_path):
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except TypeError:
            print('fail')
            return
        self.add(img)

    def _compute_image_from_path(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.mean(axis=(0,1)), img.std(axis=(0,1)), img.shape

    def _add_image_from_stats(self, image_mean, image_std, image_shape):
        self.all_mean.append(image_mean)
        self.all_std.append(image_std)
        self.Ni += 1

    def add_img_from_dir(self, image_dir, nbr_workers = 1, image_types = ['png', 'jpg', 'jpeg']):
        image_types = set(['.' + t.lower() for t in image_types])
        if nbr_workers > 1:
            result = []
            with mp.Pool(processes=nbr_workers) as pool:
                for img_path in os.listdir(image_dir):
                    if osp.splitext(img_path)[1].lower() in image_types:
                        r = pool.apply_async(self._compute_image_from_path, (osp.join(image_dir, img_path),))
                        result.append(r)
                for r in tqdm(result):
                    try:
                        self._add_image_from_stats(*r.get())
                    except TypeError:
                        print('fail')
                        pass

        else:
            for img_path in tqdm(os.listdir(image_dir)):
                if osp.splitext(img_path)[1].lower() in image_types:
                    self.add_img_from_path(osp.join(image_dir, img_path))

    def _compute(self):
        means = np.array(self.all_mean)
        stds = np.array(self.all_std)
        w = 1/self.Ni
        self.final_mean = means.mean(axis=0)
        self.final_std = np.sqrt(w*(np.sum(stds**2, axis=0) + np.sum(means**2, axis=0)) - self.final_mean**2)

    def write_yaml(self, yaml_path):
        self._compute()
        yaml_dict = {
            'PIXEL_MEAN': self.final_mean.tolist(),
            'PIXEL_STD': self.final_std.tolist()
        }
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(yaml_dict, f)
        return yaml_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate mean and std for images')
    parser.add_argument('img_dir', type=str, help='Path to images')
    parser.add_argument('--out', type=str, default = 'stats.yaml', help='Path to output yaml')
    parser.add_argument('--types', nargs = '+', type=str, default = ['png', 'jpg', 'jpeg'], help='Image type')
    parser.add_argument('-j', '--nbr-workers', type=int, default = 1, help='Number of processes to split work on')

    args = parser.parse_args()

    t = time.time()
    stats = ImageStats()
    stats.add_img_from_dir(args.img_dir, nbr_workers = args.nbr_workers, image_types = args.types)
    result = stats.write_yaml(args.out)
    print('Took {}s'.format(time.time()-t))
    print(result)
