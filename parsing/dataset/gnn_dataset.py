from typing import List
import os.path as osp
import os
import json
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import numpy as np
import networkx as nx
import time
from copy import deepcopy
from PIL import Image
from skimage import io
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
# from parsing.gnn import get_line_graph_edge_idx

def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


class Annotations:
    def __init__(self, ann_file, attr_only=False):
        if osp.splitext(ann_file)[1] == '.json':
            self.use_h5 = False
            with open(ann_file,'r') as _:
                self.annotations = json.load(_)
            self.annotations = [a for a in self.annotations if len(a['edges_negative']) > 0]
        else:
            self.attr_only = attr_only
            self.use_h5 = True
            self.ann_file = ann_file
            self.h5 = None
            self.h5_keys = []
            with h5py.File(ann_file, 'r') as f:
                for k in f.keys():
                    grp = f.get(k)
                    if len(grp.get('edges_negative')) > 0:
                        self.h5_keys.append(k)


    def __getitem__(self, idx):
        if self.use_h5:
            return self._get_h5(idx)
        else:
            return self._get_json(idx)

    def __len__(self):
        if self.use_h5:
            return len(self.h5_keys)
        else:
            return len(self.annotations)

    def _get_json(self, idx):
        ann = self.annotations[idx]
        ann = deepcopy(ann)
        for key,_type in (['junctions',np.float32],
                          ['junctions_semantic',np.long],
                          ['edges_positive',np.long],
                          ['edges_negative',np.long],
                          ['edges_semantic',np.long]):


            ann[key] = np.array(ann[key],dtype=_type)
        return ann

    def _get_h5(self, idx):
        if self.h5 is None:
            self.h5 = h5py.File(self.ann_file, 'r')
        grp = self.h5.get(self.h5_keys[idx])

        ann = {k:grp.attrs[k] for k in ['filename', 'height', 'width']}

        if not self.attr_only:
            for k in grp.keys():
                ann[k] = grp.get(k)

        return ann



class WireframeGNNDataset(Dataset):
    def __init__(self, root, gnn_root, ann_file, ann_transform = None, transform=None, training = False):
        self.gnn_root = gnn_root
        self.annotations = Annotations(ann_file, attr_only = training)

        self.transform = transform
        self.training = training
        self.ann_transform = ann_transform

        super().__init__(root, transform, None)


    @property
    def processed_dir(self) -> str:
        return osp.join(self.gnn_root, 'gnn_processed')

    def _process(self):
        os.makedirs(self.processed_dir, exist_ok = True)

        for ann in tqdm(self.annotations):
            if self.file_processed(ann['filename']):
                continue
            else:
                self.process(ann)

    def adjacent_matrix(self, n, edges, labels = None):
        dtype = torch.bool if labels is None else torch.long
        mat = torch.zeros(n+1,n+1,dtype=dtype)
        if edges.size(0)>0:
            val = True if labels is None else labels
            mat[edges[:,0], edges[:,1]] = val
            mat[edges[:,1], edges[:,0]] = val

        return mat


    def file_processed(self, img_filename):
        return (
            osp.exists(self._get_processed_fname(img_filename, train=True))
            and osp.exists(self._get_processed_fname(img_filename, train=False))
            )

    def len(self):
        return len(self.annotations)

    def process(self, ann):

        if self.ann_transform:
            ann = self.ann_transform(ann)

        npz_filename = ann['filename'] + '.npz'
        line_npz = np.load(osp.join(self.gnn_root, npz_filename))


        # To torch so that we can copy code from hafm.py
        junctions = torch.from_numpy(ann['junctions'])
        junction_labels = torch.from_numpy(ann['junctions_semantic'])
        edges_positive = torch.from_numpy(ann['edges_positive'])
        lpos_labels = torch.from_numpy(ann['edges_semantic'])
        line2junc_idx = torch.from_numpy(line_npz['junction_idx'])
        juncs_pred = torch.from_numpy(line_npz['junction_coordinates'])
        N = juncs_pred.size(0)

        train_ann = {}
        val_ann = {}
        sx = ann['width']/128.0
        sy = ann['height']/128.0
        sz = (sx+sy)/2.0
        pos_mat = self.adjacent_matrix(junctions.size(0),edges_positive,labels=lpos_labels)
        cost_, match_ = torch.sum((juncs_pred-junctions[:,None])**2,dim=-1).min(0)
        invalid_match_mask = cost_>(1.5*sz)**2  #Increase from 1.5 to 3 to get some correct lines to work with.
        target_junction_labels = torch.zeros(N, dtype=junction_labels.dtype)
        target_junction_labels[~invalid_match_mask] = junction_labels[match_[~invalid_match_mask]]
        train_ann['gnn_target_junction_labels'] =target_junction_labels

        match_[invalid_match_mask] = junctions.size(0)
        gnn_target_line_labels  = pos_mat[match_[line2junc_idx[:,0]],match_[line2junc_idx[:,1]]]

        #MASK FOR DEBUG
        # debug_mask = gnn_target_line_labels>0

        # Draw some false samples
        # nbr_false = np.count_nonzero(debug_mask) + 5
        # false_idx = np.flatnonzero(~debug_mask)
        # sample_false_idx = false_idx[np.random.randint(0,len(false_idx)-1,nbr_false)]
        # debug_mask[sample_false_idx] = True

        # junction_idx = line_npz['junction_idx'][debug_mask]
        # line_coordinates = line_npz['line_coordinates'][debug_mask]
        # line_features = line_npz['line_features'][debug_mask]
        # line2junc_idx = line2junc_idx[debug_mask]
        # gnn_target_line_labels = gnn_target_line_labels[debug_mask]
        junction_idx = line_npz['junction_idx']
        line_coordinates = line_npz['line_coordinates']
        line_features = line_npz['line_features']

        edge2idx_mat = -np.ones([N,N], dtype=np.long)
        edge2idx_mat[junction_idx[:,0],junction_idx[:,1]] = np.arange(junction_idx.shape[0])
        edge2idx_mat[junction_idx[:,1],junction_idx[:,0]] = np.arange(junction_idx.shape[0])

        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_edges_from(junction_idx.tolist())

        L = nx.line_graph(G)
        junc2line_idx = []

        #Array features according to new order
        edge2Lnode = [edge2idx_mat[n[0],n[1]] for n in L.nodes()]
        gnn_line_features = torch.from_numpy(line_features)
        line_coordinates = torch.from_numpy(line_coordinates)
        Lnode2idx = {frozenset(n): edge2idx_mat[n[0],n[1]] for n in L.nodes()}


        edge_index = []
        for e in L.edges(data=False):
            n1 = Lnode2idx[frozenset(e[0])]
            n2 = Lnode2idx[frozenset(e[1])]
            edge_index.append((n1,n2))

        # DEBUG
        # import seaborn as sns
        # LINE_COLORS = sns.color_palette('bright',6)
        # S_original = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        # S_original.sort(key= lambda g: -g.number_of_edges())
        # print('Connected subgraphs', len(S_original))
        # image = Image.open(osp.join(self.root,ann['filename'])).convert('RGB')
        # # sx = image.width/128.0
        # # sy = image.height/128.0
        # line_coordinates_plot = line_coordinates.clone().numpy()
        # # line_coordinates_plot[:,::2] *= sx
        # # line_coordinates_plot[:,1::2] *= sy
        # i_max = min(3,len(S_original))
        # plt.figure()
        # plt.subplot(2,2,1)
        # plt.imshow(image)
        # plt.plot([line_coordinates_plot[:,0], line_coordinates_plot[:,2]],
        #         [line_coordinates_plot[:,1], line_coordinates_plot[:,3]],
        #         linewidth = 3)
        # for i in range(i_max):
        #     plt.subplot(2,2,i+2)
        #     plt.imshow(image)
        #     sub_idx = [edge2idx_mat[e[0],e[1]] for e in S_original[i].edges()]
        #     lines = line_coordinates_plot[sub_idx]
        #     labels = gnn_target_line_labels[sub_idx]
        #     for l in labels.unique():
        #         lines_l = lines[l==labels]
        #         plt.plot([lines_l[:,0], lines_l[:,2]],
        #                 [lines_l[:,1], lines_l[:,3]],
        #                 linewidth = 3,
        #                 color = LINE_COLORS[l])
        #     plt.title(str(len(sub_idx)))
        # plt.savefig('/host_home/debug/line_graph/original_{}'.format(ann['filename']))
        # plt.close()
        #
        #
        # S = [L.subgraph(c).copy() for c in nx.connected_components(L)]
        # S.sort(key= lambda g: -g.number_of_nodes())
        # print('L subgraphs', len(S))
        # i_max = min(3,len(S))
        # plt.figure()
        # plt.subplot(2,2,1)
        # plt.imshow(image)
        # plt.plot([line_coordinates_plot[:,0], line_coordinates_plot[:,2]],
        #         [line_coordinates_plot[:,1], line_coordinates_plot[:,3]],
        #         linewidth = 3)
        # for i in range(i_max):
        #     plt.subplot(2,2,i+2)
        #     plt.imshow(image)
        #     sub_idx = [Lnode2idx[frozenset(n)] for n in S[i].nodes()]
        #     lines = line_coordinates_plot[sub_idx]
        #     labels = gnn_target_line_labels[sub_idx]
        #     for l in labels.unique():
        #         lines_l = lines[l==labels]
        #         plt.plot([lines_l[:,0], lines_l[:,2]],
        #                 [lines_l[:,1], lines_l[:,3]],
        #                 linewidth = 3,
        #                 color = LINE_COLORS[l])
        #     plt.title(str(len(sub_idx)))
        # plt.savefig('/host_home/debug/line_graph/line_graph_{}'.format(ann['filename']))
        # plt.close()
        # for so,sn in zip(S_original,S):
        #     if so.number_of_edges() > 0:
        #         if so.number_of_edges() != sn.number_of_nodes():
        #             print(so.number_of_edges() ,sn.number_of_nodes())
        #         assert so.number_of_edges() == sn.number_of_nodes()
        #
        # G2 = nx.Graph()
        # G2.add_nodes_from(range(line_coordinates.shape[0]))
        # G2.add_edges_from(edge_index)
        # S = [G2.subgraph(c).copy() for c in nx.connected_components(G2)]
        # S.sort(key= lambda g: -g.number_of_nodes())
        # print('G2 subgraphs', len(S))
        # i_max = min(3,len(S))
        # plt.figure()
        # plt.subplot(2,2,1)
        # plt.imshow(image)
        # plt.plot([line_coordinates_plot[:,0], line_coordinates_plot[:,2]],
        #         [line_coordinates_plot[:,1], line_coordinates_plot[:,3]],
        #         linewidth = 3)
        # for i in range(i_max):
        #     plt.subplot(2,2,i+2)
        #     plt.imshow(image)
        #     sub_idx = [n for n in S[i].nodes()]
        #     lines = line_coordinates_plot[sub_idx]
        #     labels = gnn_target_line_labels[sub_idx]
        #     for l in labels.unique():
        #         lines_l = lines[l==labels]
        #         plt.plot([lines_l[:,0], lines_l[:,2]],
        #                 [lines_l[:,1], lines_l[:,3]],
        #                 linewidth = 3,
        #                 color = LINE_COLORS[l])
        #     plt.title(str(len(sub_idx)))
        # plt.savefig('/host_home/debug/line_graph/line_graph2_{}'.format(ann['filename']))
        # plt.close()
        # for so,sn in zip(S_original,S):
        #     if so.number_of_edges() > 0:
        #         assert so.number_of_edges() == sn.number_of_nodes()
        # sys.exit()
        # DEBUG END


        val_ann['prior_junctions'] = juncs_pred
        val_ann['prior_lines'] = line_coordinates
        val_ann['prior_line2junc_idx'] = line2junc_idx

        train_ann['gnn_target_line_labels'] = gnn_target_line_labels

        train_ann['data_line'] = Data(x = gnn_line_features,
                        edge_index = torch.tensor(edge_index, dtype=torch.long).T)

        gnn_junction_features = torch.tensor(line_npz['junction_features'], dtype=torch.float32).T
        train_ann['data_junction'] = Data(x = gnn_junction_features,
                        edge_index = line2junc_idx.T)

        torch.save(train_ann, self._get_processed_fname(ann['filename'], train=True))
        torch.save(val_ann, self._get_processed_fname(ann['filename'], train=False))

    def _get_processed_fname(self, img_fname, train=True):
        train_val_str = 'train' if train else 'val'
        return osp.join(self.processed_dir, '{}.{}.pt'.format(img_fname,train_val_str))

    def get_train(self, idx):
        ann = self.annotations[idx]
        train_ann = torch.load(self._get_processed_fname(ann['filename'], train=True))
        data = (train_ann['data_line'],train_ann['data_junction'])
        del train_ann['data_line']
        del train_ann['data_junction']

        if self.transform:
            return self.transform(data,train_ann)


        return data, train_ann

    def get_val(self, idx):
        ann = self.annotations[idx]
        ann.update( torch.load(self._get_processed_fname(ann['filename'], train=True)) )
        ann.update( torch.load(self._get_processed_fname(ann['filename'], train=False)) )
        data = (ann['data_line'],ann['data_junction'])
        del ann['data_line']
        del ann['data_junction']

        if self.transform:
            return self.transform(data,ann)

        return data, ann

    def get(self, idx):
        if self.training:
            return self.get_train(idx)
        else:
            return self.get_val(idx)


    def image(self, idx):
        ann = self.annotations[idx]
        image = Image.open(osp.join(self.root,ann['filename'])).convert('RGB')
        return image


class WireframeGNNDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collator = self.collate_fn
        self.collate_fn = lambda x: _wireframe_collate(x,self.collator)

def ann_collator(annotations):
    if len(annotations) == 1:
        return annotations[0]
    col_ann = {}
    for key in annotations[0].keys():
        col_ann[key] = torch.cat([a[key] for a in annotations], dim = 0)
    return col_ann


def _wireframe_collate(batch, data_collator):
    return (data_collator([b[0] for b in batch]),
            ann_collator([b[1] for b in batch]))
