import torch
import torchvision
from torchvision.transforms import functional as F
from skimage.transform  import resize
from parsing.utils.labels import LabelMapper

import numpy as np
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, ann=None):
        if ann is None:
            for t in self.transforms:
                image = t(image)
            return image
        for t in self.transforms:
            image, ann = t(image, ann)
        return image, ann
class Resize(object):
    def __init__(self, image_height, image_width, ann_height, ann_width):
        self.image_height = image_height
        self.image_width  = image_width
        self.ann_height   = ann_height
        self.ann_width    = ann_width

    def __call__(self, image, ann):
        image = resize(image,(self.image_height,self.image_width))
        image = np.array(image,dtype=np.float32)/255.0

        sx = self.ann_width/ann['width']
        sy = self.ann_height/ann['height']
        ann['junctions'][:,0] = np.clip(ann['junctions'][:,0]*sx, 0, self.ann_width-1e-4)
        ann['junctions'][:,1] = np.clip(ann['junctions'][:,1]*sy, 0, self.ann_height-1e-4)
        ann['width'] = self.ann_width
        ann['height'] = self.ann_height

        return image, ann

class ResizeAnn(object):
    def __init__(self, ann_height, ann_width):
        self.ann_height   = ann_height
        self.ann_width    = ann_width

    def __call__(self, ann):
        sx = self.ann_width/ann['width']
        sy = self.ann_height/ann['height']
        ann['junctions'][:,0] = np.clip(ann['junctions'][:,0]*sx, 0, self.ann_width-1e-4)
        ann['junctions'][:,1] = np.clip(ann['junctions'][:,1]*sy, 0, self.ann_height-1e-4)
        ann['width'] = self.ann_width
        ann['height'] = self.ann_height

        return ann


class ResizeImage(object):
    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width  = image_width

    def __call__(self, image, ann=None):
        image = resize(image,(self.image_height,self.image_width))
        image = np.array(image,dtype=np.float32)/255.0
        if ann is None:
            return image
        return image, ann

class ResizeGNN(object):
    def __init__(self, ann_height, ann_width):
        self.ann_height   = ann_height
        self.ann_width    = ann_width

    def __call__(self, data, ann):

        sx = self.ann_width/ann['width']
        sy = self.ann_height/ann['height']
        ann['junctions'][:,0] = np.clip(ann['junctions'][:,0]*sx, 0, self.ann_width-1e-4)
        ann['junctions'][:,1] = np.clip(ann['junctions'][:,1]*sy, 0, self.ann_height-1e-4)
        ann['width'] = self.ann_width
        ann['height'] = self.ann_height

        return data, ann

#Noop for now
class ReMapLabels(object):
    def __init__(self, line_type='single', disable=False):
        self.lm = LabelMapper(None, None, disable=disable)
        self.disable = disable

    def __call__(self, image, anns=None):
        if anns is None:
            return image

        anns['edges_semantic'] = self.lm.map_lines(anns['edges_semantic'])
        try:
            ann['gnn_edges_semantic'] = self.lm.map_lines(anns['gnn_edges_semantic'])
        except KeyError:
            pass

        anns['junctions_semantic'] = self.lm.map_junctions(anns['junctions_semantic'])
        # if self.disable:
        #     anns['junc_occluded'][:] = True

        return image, anns

class ToTensor(object):
    def __call__(self, image, anns=None):
        if anns is None:
            return F.to_tensor(image)

        for key,val in anns.items():
            if isinstance(val,np.ndarray):
                anns[key] = torch.from_numpy(val)
        return F.to_tensor(image),anns

class ToTensorGNN(object):
    def __call__(self, data, anns=None):
        if anns is None:
            return data

        for key,val in anns.items():
            if isinstance(val,np.ndarray):
                anns[key] = torch.from_numpy(val)
        return data,anns



class Normalize(object):
    def __init__(self, mean, std,to_255 = False):
        self.mean = mean
        self.std  = std
        self.to_255 = to_255
    def __call__(self, image, anns = None):
        if self.to_255:
            image*=255.0
        image = F.normalize(image,mean=self.mean,std=self.std)
        if anns is None:
            return image
        return image, anns
