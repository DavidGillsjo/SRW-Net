import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
import networkx as nx
from parsing.utils.labels import LabelMapper
import time

def convert_to_line_graph_edge_idx(junction_idx, num_junc):
    edge2idx_mat = -torch.ones([num_junc,num_junc], dtype=torch.long)
    edge2idx_mat[junction_idx[:,0],junction_idx[:,1]] = torch.arange(junction_idx.shape[0])
    edge2idx_mat[junction_idx[:,1],junction_idx[:,0]] = torch.arange(junction_idx.shape[0])

    # try:
    #     junction_idx = junction_idx.tolist()
    # except AttributeError:
    #     pass
    t = time.time()
    G = nx.Graph()
    G.add_nodes_from(range(num_junc))
    G.add_edges_from(junction_idx.tolist())
    t = time.time()
    L = nx.line_graph(G)
    junc2line_idx = []

    #Array features according to new order
    edge2Lnode = [edge2idx_mat[n[0],n[1]] for n in L.nodes()]
    Lnode2idx = {frozenset(n): edge2idx_mat[n[0],n[1]] for n in L.nodes()}


    edge_index = []
    for e in L.edges(data=False):
        n1 = Lnode2idx[frozenset(e[0])]
        n2 = Lnode2idx[frozenset(e[1])]
        edge_index.append((n1,n2))

    edge_index = torch.tensor(edge_index, dtype=torch.long, device=junction_idx.device).T
    print('Networkx line graph',  time.time() - t)

    return edge_index

def torch_convert_to_line_graph_edge_idx(junction_idx, num_junc):
    t = time.time()
    # Unidirectional
    pt_data = Data(edge_index = torch.cat([junction_idx.T,torch.flip(junction_idx.T,[0])],dim=1),
                   num_nodes = num_junc,
                   edge_attr = torch.cat([torch.arange(junction_idx.size(0),device = junction_idx.device)]*2)
                   )
    tform = LineGraph()
    data = tform(pt_data)
    #Assert order is preserved, should be since we sort during detection
    assert torch.all(torch.bitwise_right_shift(data.x,1) == torch.arange(junction_idx.size(0), device = data.x.device))

    return data.edge_index

def make_unidirectional(junction_idx, num_junc):
    data = Data(edge_index = torch.cat([junction_idx.T,torch.flip(junction_idx.T,[0])],dim=1),
                   num_nodes = num_junc)
    data.coalesce()
    return data.edge_index

class GNNLayer(nn.Module):
    def __init__(self, input_dim ,output_dim):
        super().__init__()
        # self.dropout = nn.Dropout(p=0.2)
        self.conv = GCNConv(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.normalization = BatchNorm(output_dim)

    def forward(self, x, edge_index, edge_features = None):
        # x = self.dropout(x)
        x = self.conv(x, edge_index)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class WireframeGNNIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,  node_features, edge_idx):
        return node_features

class WireframeGNNIdentityBatchFriendly(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        return data.x

"""
### Takes a set of junction and line features with connections indicated by a list of junction idx.
### Converts this to a graph representation with lines as nodes and junctions as edges.
### Performs convolutions and returns the updated features.
"""
class WireframeGNN(nn.Module):
    def __init__(self, line_feature_len, n_layers = 2):
        super().__init__()
        self.layers = []
        self.out_channels = line_feature_len
        current_input_dim = line_feature_len
        for i in range(n_layers):
            self.layers.append(
                GNNLayer(current_input_dim, current_input_dim//2)
            )
            current_input_dim = current_input_dim//2
            self.out_channels += current_input_dim
        self.layers = nn.ModuleList(self.layers)
        self.fc = nn.Sequential(
            nn.Linear(self.out_channels, line_feature_len),
            nn.ReLU()
            )


    def forward(self, node_features, edge_idx):
        features = [node_features]
        for l in self.layers:
            node_features = l(node_features, edge_idx)
            features.append(node_features)
        features = torch.cat(features, dim=1)
        features = self.fc(features)
        return features

class WireframeGNNBatchFriendly(WireframeGNN):

    def forward(self, data):
        line_features = data.x
        features = [data.x]
        for l in self.layers:
            line_features = l(line_features, data.edge_index, data.edge_attr)
            features.append(line_features)
        features = torch.cat(features, dim=1)
        features = self.fc(features)
        return features


"""
### WireframeGNN head for end-to-end training
"""
class WireframeGNNHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_pts1     = cfg.MODEL.PARSING_HEAD.N_PTS1
        self.dim_loi    = cfg.MODEL.PARSING_HEAD.DIM_LOI
        self.dim_fc     = cfg.MODEL.PARSING_HEAD.DIM_FC
        self.num_feats   = cfg.MODEL.HGNETS.NUM_FEATS
        label_mapper = LabelMapper(cfg.MODEL.LINE_LABELS, cfg.MODEL.JUNCTION_LABELS, disable=cfg.DATASETS.DISABLE_CLASSES)
        self.nbr_line_labels = label_mapper.nbr_line_labels()
        self.nbr_junction_labels = len(cfg.MODEL.JUNCTION_LABELS)
        if cfg.MODEL.GNN.LAYERS > 0:
            self.line_gnn = WireframeGNN(self.dim_fc, cfg.MODEL.GNN.LAYERS)
            self.line_gnn_forward = self._line_forward
        else:
            self.line_gnn_forward = self._line_bypass
        if cfg.MODEL.GNN.JUNCTION_LAYERS > 0:
            self.junction_gnn = WireframeGNN(self.num_feats, cfg.MODEL.GNN.JUNCTION_LAYERS)
            self.junction_gnn_forward = self._junction_forward
        else:
            self.junction_gnn_forward = self._junction_bypass


        last_fc = nn.Linear(self.dim_fc, self.nbr_line_labels)

        self.fc_lines = nn.Sequential(
            nn.Linear(self.dim_loi * self.n_pts1, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, self.dim_fc),
            nn.ReLU(inplace=True),
            last_fc
            )
        if cfg.MODEL.GNN.JUNCTION_LAYERS > 0:
            m = int(self.num_feats / 4)
            self.fc_junction = nn.Sequential(
                nn.Linear(self.num_feats, m),
                nn.ReLU(inplace=True),
                nn.Linear(m, self.nbr_junction_labels),
                )

    def forward(self, line_features, junction_features, line2junction_idx, junction_logits, line_logits):
        line_logits = self.line_gnn_forward(line_features, line2junction_idx, junction_features.size(0), line_logits)
        junction_logits = self.junction_gnn_forward(junction_features, line2junction_idx, junction_logits)
        return line_logits, junction_logits

    def _line_bypass(self, line_features, line2junction_idx, num_junctions, line_logits):
        return line_logits

    def _line_forward(self, line_features, line2junction_idx, num_junctions, line_logits):
        edge_idx = torch_convert_to_line_graph_edge_idx(line2junction_idx, num_junctions)
        line_features = self.line_gnn(
            line_features,
            edge_idx)
        return self.fc_lines(line_features)

    def _junction_bypass(self, junction_features, line2junction_idx, junction_logits):
        return junction_logits

    def _junction_forward(self, junction_features, line2junction_idx, junction_logits):
        edge_idx = make_unidirectional(line2junction_idx, junction_features.size(0))
        junction_features = self.junction_gnn(
            junction_features,
            edge_idx)
        return self.fc_junction(junction_features)


"""
### Standalone network trained on stored wireframes
"""
class WireframeGNNClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_pts1     = cfg.MODEL.PARSING_HEAD.N_PTS1
        self.dim_loi    = cfg.MODEL.PARSING_HEAD.DIM_LOI
        self.dim_fc     = cfg.MODEL.PARSING_HEAD.DIM_FC
        self.num_feats   = cfg.MODEL.HGNETS.NUM_FEATS
        label_mapper = LabelMapper(cfg.MODEL.LINE_LABELS, cfg.MODEL.JUNCTION_LABELS, disable=cfg.DATASETS.DISABLE_CLASSES)
        self.nbr_line_labels = label_mapper.nbr_line_labels()
        self.nbr_junction_labels = len(cfg.MODEL.JUNCTION_LABELS)
        if cfg.MODEL.GNN.LAYERS > 0:
            self.gnn = WireframeGNNBatchFriendly(self.dim_fc, cfg.MODEL.GNN.LAYERS)
        else:
            self.gnn = WireframeGNNIdentityBatchFriendly()
        if cfg.MODEL.GNN.JUNCTION_LAYERS > 0:
            self.junction_gnn = WireframeGNNBatchFriendly(self.num_feats, cfg.MODEL.GNN.JUNCTION_LAYERS)
        else:
            self.junction_gnn = WireframeGNNIdentityBatchFriendly()
        self.target_width = cfg.DATASETS.TARGET.WIDTH
        self.target_height = cfg.DATASETS.TARGET.HEIGHT

        if cfg.MODEL.LINE_LOSS_WEIGHTS:
            line_loss = torch.tensor(cfg.MODEL.LINE_LOSS_WEIGHTS, dtype=torch.float32)
        else:
            line_loss = None
        self.loss = nn.CrossEntropyLoss(reduction='mean', weight = line_loss)

        if cfg.MODEL.JUNCTION_LOSS_WEIGHTS:
            junction_label_weights = torch.tensor(cfg.MODEL.JUNCTION_LOSS_WEIGHTS, dtype=torch.float32)
        else:
            junction_label_weights = None
        self.junction_loss = nn.CrossEntropyLoss(reduction='mean',weight = junction_label_weights)

        self.FvsT_sample_ratio = cfg.MODEL.FALSE_VS_POSITIVE_SAMPLE_RATIO

        last_fc = nn.Linear(self.dim_fc, self.nbr_line_labels)

        self.fc_lines = nn.Sequential(
            nn.Linear(self.dim_loi * self.n_pts1, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, self.dim_fc),
            nn.ReLU(inplace=True),
            last_fc
            )
        m = int(self.num_feats / 4)
        self.fc_junction = nn.Sequential(
            nn.Linear(self.num_feats, m),
            nn.ReLU(inplace=True),
            nn.Linear(m, self.nbr_junction_labels),
            )

    def forward(self, data, annotations = None):
        if self.training:
            return self.forward_train(data, annotations)
        else:
            return self.forward_test(data, annotations)

    def forward_test(self, data, annotations):

        data_line, data_junction = data
        device = data_line.x.device
        extra_info = {
            'time_gnn': 0.0,
            'time_classification': 0.0,
        }

        extra_info['time_gnn'] = time.time()
        if data_line.edge_index.size(0) > 0:
            line_features = self.gnn(data_line)
        else:
            line_features = data_line.x
        if data_junction.edge_index.size(0) > 0:
            junction_features = self.junction_gnn(data_junction)
        else:
            junction_features = data_junction.x
        extra_info['time_gnn'] = time.time() - extra_info['time_gnn']

        extra_info['time_classification'] = time.time()
        logits = self.fc_lines(line_features)
        # print('ratio_neq', torch.sum(logits.argmax(1) != annotations['gnn_target_line_labels'])/len(annotations['gnn_target_line_labels']))
        scores = logits.softmax(1)
        lines_label = scores.argmax(1)
        # print('lines_label',lines_label.unique())

        junction_logits = self.fc_junction(junction_features)
        junction_scores = junction_logits.softmax(1)
        junction_label = junction_scores.argmax(1)

        junction_coordinates = annotations['prior_junctions']
        line_coordinates = annotations['prior_lines']

        assert line_coordinates.size(0) == data_line.x.size(0)
        assert junction_coordinates.size(0) == data_junction.x.size(0)

        extra_info['junc_prior_ver'] = junction_coordinates
        extra_info['lines_prior_ver'] = line_coordinates
        extra_info['lines_prior_scoring'] = line_coordinates

        lines_score_valid = 1-scores[:,0]
        valid_mask = lines_score_valid > 0.05
        lines_final = line_coordinates[valid_mask]
        lines_label = lines_label[valid_mask]
        scores = scores[valid_mask]
        lines_score_label = torch.gather(scores, 1, lines_label.unsqueeze(1)).squeeze(1)
        lines_score_valid = lines_score_valid[valid_mask]


        # TODO: Supply edges for the junctions?
        idx_lines_for_junctions = annotations['prior_line2junc_idx']
        unique_j_idx, l2j_idx = idx_lines_for_junctions[valid_mask].unique(return_inverse=True)
        juncs_final = junction_coordinates[unique_j_idx]
        juncs_score = junction_scores[unique_j_idx]
        juncs_label = junction_label[unique_j_idx]
        juncs_valid_score = 1-juncs_score[:,0]
        juncs_label_score = torch.gather(juncs_score, 1, juncs_label.unsqueeze(1)).squeeze(1)

        extra_info['time_classification'] = time.time() - extra_info['time_classification']

        output = {
            'num_proposals': 0,
            'filename': annotations['filename'] if annotations else None,
            'width': annotations['width'],
            'height': annotations['height'],
        }

        output.update({
            'lines_pred': lines_final,
            'lines_label': lines_label,
            'lines_valid_score': lines_score_valid,
            'lines_label_score': lines_score_label,
            'lines_score': scores,
            'juncs_pred': juncs_final,
            'juncs_label': juncs_label,
            'juncs_valid_score': juncs_valid_score,
            'juncs_label_score': juncs_label_score,
            'juncs_score': juncs_score,
            'line2junc_idx': l2j_idx,
            'num_proposals': line_coordinates.size(0),
        })

        return output, extra_info

    def forward_train(self, data,  annotations):
        data_line, data_junction = data
        device = data_line.x.device

        extra_info = {
            'time_gnn': 0.0,
            'time_classification': 0.0,
        }

        extra_info['time_gnn'] = time.time()
        line_features = self.gnn(data_line)
        junction_features = self.junction_gnn(data_junction)
        extra_info['time_gnn'] = time.time() - extra_info['time_gnn']

        extra_info['time_classification'] = time.time()
        line_logits = self.fc_lines(line_features)
        junction_logits = self.fc_junction(junction_features)
        extra_info['time_classification'] = time.time() - extra_info['time_classification']

        # print('ratio_neq', torch.sum(line_logits.argmax(1) != annotations['gnn_target_line_labels'])/len(annotations['gnn_target_line_labels']))
        # print('labels', annotations['gnn_target_line_labels'].unique())
        if self.FvsT_sample_ratio is None:
            loss_dict = {
                'loss_line_label': self.loss(line_logits, annotations['gnn_target_line_labels']),
                'loss_junction_label': self.junction_loss(junction_logits, annotations['gnn_target_junction_labels'])
            }
        else:
            sample_idx = {}
            for k in ['gnn_target_line_labels', 'gnn_target_junction_labels']:
                true_mask = (annotations[k] > 0)
                true_idx = torch.nonzero(true_mask,as_tuple=False).squeeze()
                nbr_true = len(true_idx)
                nbr_false_to_sample = int(nbr_true*self.FvsT_sample_ratio)
                false_idx = torch.nonzero(~true_mask,as_tuple=False).squeeze()
                nbr_false = len(false_idx)
                if nbr_false_to_sample < nbr_false:
                    false_idx_to_sample = false_idx[torch.randperm(nbr_false)[:nbr_false_to_sample]]
                    sample_idx[k] = torch.cat((true_idx, false_idx_to_sample))
                else:
                    sample_idx[k] = torch.arange(len(annotations[k]))

            loss_dict = {
                'loss_line_label': self.loss(line_logits[sample_idx['gnn_target_line_labels']],
                                             annotations['gnn_target_line_labels'][sample_idx['gnn_target_line_labels']]),
                'loss_junction_label': self.junction_loss(junction_logits[sample_idx['gnn_target_junction_labels']],
                                                          annotations['gnn_target_junction_labels'][sample_idx['gnn_target_junction_labels']])
            }

        return loss_dict, extra_info
