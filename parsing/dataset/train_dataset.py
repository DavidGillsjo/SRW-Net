import torch
from torch.utils.data import Dataset

import os.path as osp
import json
import cv2
from skimage import io
from PIL import Image
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import copy
import os
class TrainDataset(Dataset):
    def __init__(self, root, ann_file, transform = None, vflip = False, hflip = False):
        self.root = root
        with open(ann_file,'r') as _:
            self.annotations = json.load(_)

        # Make sure there is at least one negative edge for each image
        self.annotations = [a for a in self.annotations if a['edges_negative']]
        self.transform = transform
        self.vflip = vflip
        self.hflip = hflip

        ann_folder = osp.dirname(ann_file)
        self.dbg_dir_prior = osp.join(ann_folder, 'train_dbg', 'prior_transform')
        self.dbg_dir_post = osp.join(ann_folder, 'train_dbg', 'post_transform')
        os.makedirs(self.dbg_dir_prior, exist_ok=True)
        os.makedirs(self.dbg_dir_post, exist_ok=True)


    def __getitem__(self, idx):
        ann = copy.deepcopy(self.annotations[idx])
        image = io.imread(osp.join(self.root,ann['filename'])).astype(float)[:,:,:3]
        for key,_type in (['junctions',np.float32],
                          ['junctions_semantic',np.long],
                          ['edges_positive',np.long],
                          ['edges_negative',np.long],
                          ['edges_semantic',np.long]):


            ann[key] = np.array(ann[key],dtype=_type)

        if not 'junctions_semantic' in ann:
            ann['junc_occluded'] = np.array(ann['junc_occluded'],dtype=np.bool)

        assert np.all(ann['edges_semantic'] > 0) #Assumes that the dataset also has an invalid class as 0

        width = ann['width']
        height = ann['height']
        #Randomize flip
        if self.hflip and random.getrandbits(1):
            image = image[:,::-1,:]
            ann['junctions'][:,0] = width-ann['junctions'][:,0]
        if self.vflip and random.getrandbits(1):
            image = image[::-1,:,:]
            ann['junctions'][:,1] = height-ann['junctions'][:,1]


        if self.transform is not None:
            image, ann = self.transform(image,ann)

        return image, ann

    def __len__(self):
        return len(self.annotations)

def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            [b[1] for b in batch])
