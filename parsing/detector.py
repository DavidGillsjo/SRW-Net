import torch
from torch import nn
from parsing.backbones import build_backbone
from parsing.encoder.hafm import HAFMencoder
# from epnet.structures.linelist_ops import linesegment_distance
import torch.nn.functional as F
import matplotlib.pyplot as plt
import  numpy as np
import time
from parsing.utils.labels import LabelMapper
from parsing.gnn import WireframeGNNHead
import random


def sigmoid_l1_loss(logits, targets, offset = 0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp-targets)

    if mask is not None:
        w = mask.mean(3, True).mean(2,True)
        w[w==0] = 1
        loss = loss*(mask/w)

    return loss.mean()

def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask

def get_junctions(jloc, joff, topk = 300, th=0):
    height, width = jloc.size(1), jloc.size(2)
    jloc_flat = jloc.flatten()
    joff_flat = joff.flatten(start_dim=1)

    scores, index = torch.topk(jloc_flat, k=topk)
    y = (index / width).float() + torch.gather(joff_flat[1], 0, index) + 0.5
    x = (index % width).float() + torch.gather(joff_flat[0], 0, index) + 0.5

    junctions = torch.stack((x, y)).t()

    score_mask = scores>th

    return junctions[score_mask], scores[score_mask], index[score_mask]

class WireframeDetector(nn.Module):
    def __init__(self, cfg):
        super(WireframeDetector, self).__init__()
        self.hafm_encoder = HAFMencoder(cfg)
        self.backbone = build_backbone(cfg)

        self.n_dyn_junc = cfg.MODEL.PARSING_HEAD.N_DYN_JUNC
        self.n_dyn_posl = cfg.MODEL.PARSING_HEAD.N_DYN_POSL
        self.n_dyn_negl = cfg.MODEL.PARSING_HEAD.N_DYN_NEGL
        self.n_dyn_othr = cfg.MODEL.PARSING_HEAD.N_DYN_OTHR
        self.n_dyn_othr2= cfg.MODEL.PARSING_HEAD.N_DYN_OTHR2
        self.n_pts0     = cfg.MODEL.PARSING_HEAD.N_PTS0
        self.n_pts1     = cfg.MODEL.PARSING_HEAD.N_PTS1
        self.dim_loi    = cfg.MODEL.PARSING_HEAD.DIM_LOI
        self.dim_fc     = cfg.MODEL.PARSING_HEAD.DIM_FC
        self.n_out_junc = cfg.MODEL.PARSING_HEAD.N_OUT_JUNC
        self.n_out_line = cfg.MODEL.PARSING_HEAD.N_OUT_LINE
        self.max_distance = cfg.MODEL.PARSING_HEAD.MAX_DISTANCE
        if self.max_distance <= 0:
            self.max_distance = float('inf')
        self.use_residual = cfg.MODEL.PARSING_HEAD.USE_RESIDUAL
        self.use_gt_junctions = cfg.MODEL.USE_GT_JUNCTIONS
        self.use_gt_lines = cfg.MODEL.USE_GT_LINES
        self.require_valid_junctions = cfg.MODEL.PARSING_HEAD.REQUIRE_VALID_JUNCTIONS
        self.output_idx = np.cumsum([0] + [h[0] for h in cfg.MODEL.HEAD_SIZE])
        label_mapper = LabelMapper(cfg.MODEL.LINE_LABELS, cfg.MODEL.JUNCTION_LABELS, disable=cfg.DATASETS.DISABLE_CLASSES)
        self.nbr_line_labels = label_mapper.nbr_line_labels()
        self.gnn_head = WireframeGNNHead(cfg)

        self.register_buffer('tspan', torch.linspace(0, 1, self.n_pts0)[None,None,:])

        if getattr(cfg.MODEL, 'LINE_LOSS_WEIGHTS', None):
            line_loss = torch.tensor(cfg.MODEL.LINE_LOSS_WEIGHTS, dtype=torch.float32)
        else:
            line_loss = None
        self.loss = nn.CrossEntropyLoss(reduction='none', weight = line_loss)

        if getattr(cfg.MODEL, 'JUNCTION_LOSS_WEIGHTS', None):
            junction_label_weights = torch.tensor(cfg.MODEL.JUNCTION_LOSS_WEIGHTS, dtype=torch.float32)
        else:
            junction_label_weights = None
        self.junction_label_loss = nn.CrossEntropyLoss(weight = junction_label_weights)
        self.gnn_junction_label_loss = nn.CrossEntropyLoss()

        self.fc1 = nn.Conv2d(256, self.dim_loi, 1)
        self.pool1d = nn.MaxPool1d(self.n_pts0//self.n_pts1, self.n_pts0//self.n_pts1)
        line_bias = getattr(cfg.MODEL, 'LINE_CLASS_BIAS', None)
        if line_bias:
            last_fc.bias.weight = torch.tensor(line_bias, dtype=torch.float32)

        last_fc = nn.Linear(self.dim_fc, self.nbr_line_labels)

        self.fc2 = nn.Sequential(
            nn.Linear(self.dim_loi * self.n_pts1, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, self.dim_fc),
            nn.ReLU(inplace=True),
            last_fc
            )

        self.train_step = 0

    def pooling(self, features_per_image, lines_per_im):
        h,w = features_per_image.size(1), features_per_image.size(2)
        U,V = lines_per_im[:,:2], lines_per_im[:,2:]
        sampled_points = U[:,:,None]*self.tspan + V[:,:,None]*(1-self.tspan) -0.5
        sampled_points = sampled_points.permute((0,2,1)).reshape(-1,2)
        px,py = sampled_points[:,0],sampled_points[:,1]
        px0 = px.floor().clamp(min=0, max=w-1)
        py0 = py.floor().clamp(min=0, max=h-1)
        px1 = (px0 + 1).clamp(min=0, max=w-1)
        py1 = (py0 + 1).clamp(min=0, max=h-1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

        xp = ((features_per_image[:, py0l, px0l] * (py1-py) * (px1 - px)+ features_per_image[:, py1l, px0l] * (py - py0) * (px1 - px)+ features_per_image[:, py0l, px1l] * (py1 - py) * (px - px0)+ features_per_image[:, py1l, px1l] * (py - py0) * (px - px0)).reshape(128,-1,32)
        ).permute(1,0,2)


        # if self.pool1d is not None:
        xp = self.pool1d(xp)
        features_per_line = xp.view(-1, self.n_pts1*self.dim_loi)
        # features_per_line = self.fc2(features_per_line)

        return features_per_line


    def _get_output_dist(self, output):
        md_out = output[:,self.output_idx[0]:self.output_idx[1]]
        dis_out = output[:,self.output_idx[1]:self.output_idx[2]]
        res_out = output[:,self.output_idx[2]:self.output_idx[3]]
        jlabel_out= output[:,self.output_idx[3]:self.output_idx[4]]
        joff_out= output[:,self.output_idx[4]:self.output_idx[5]]
        return md_out, dis_out, res_out, jlabel_out, joff_out

    def forward(self, images, annotations = None, output_features = False):
        if self.training:
            return self.forward_train(images, annotations=annotations)
        else:
            return self.forward_test(images, annotations=annotations, output_features=output_features)

    def forward_test(self, images, annotations = None, output_features = False):
        device = images.device

        extra_info = {
            'time_backbone': 0.0,
            'time_proposal': 0.0,
            'time_matching': 0.0,
            'time_verification': 0.0,
        }

        extra_info['time_backbone'] = time.time()
        outputs, features = self.backbone(images)

        loi_features = self.fc1(features)
        md_out, dis_out, res_out, jlabel_out, joff_out = self._get_output_dist(outputs[0])
        md_pred = md_out.sigmoid()
        dis_pred = dis_out.sigmoid()
        res_pred = res_out.sigmoid()
        jlabel_prob = jlabel_out.softmax(1)
        jloc_pred = 1-jlabel_prob[:,0,None]
        joff_pred= joff_out.sigmoid() - 0.5
        extra_info['time_backbone'] = time.time() - extra_info['time_backbone']
        # Extra info for plotting intermediate results
        # extra_info['jloc_pred'] = jloc_pred


        batch_size = md_pred.size(0)
        assert batch_size == 1
        ann = annotations[0]

        extra_info['time_proposal'] = time.time()
        if self.use_gt_lines:
            junctions = ann['junctions']
            junctions[:,0] *= 128/float(ann['width'])
            junctions[:,1] *= 128/float(ann['height'])
            edges_positive = ann['edges_positive']
            lines_pred = torch.cat((junctions[edges_positive[:,0]], junctions[edges_positive[:,1]]),dim=-1).to(device)
        elif self.use_residual:
            lines_pred = self.proposal_lines_new(md_pred[0],dis_pred[0],res_pred[0]).view(-1,4)
        else:
            lines_pred = self.proposal_lines_new(md_pred[0], dis_pred[0], None).view(-1, 4)

        jloc_pred_nms = non_maximum_suppression(jloc_pred[0])
        topK = min(self.n_out_junc, int((jloc_pred_nms>0.008).float().sum().item()))

        if self.use_gt_junctions:
            juncs_pred = ann['junctions'].to(device)
            juncs_pred[:,0] *= 128/float(ann['width'])
            juncs_pred[:,1] *= 128/float(ann['height'])
            juncs_label = ann['junctions_semantic']
            juncs_score = torch.zeros([juncs_pred.size(0), jlabel_prob.size(1)])
            juncs_score[range(juncs_label.size(0)), juncs_label] = 1
            juncs_logits = juncs_score
        else:
            juncs_pred, juncs_valid_score, flat_index = get_junctions(jloc_pred_nms, joff_pred[0], topk=topK)
            juncs_logits = (jlabel_out.flatten(start_dim=2)[0,:,flat_index]).T
            juncs_score = (jlabel_prob.flatten(start_dim=2)[0,:,flat_index]).T
            juncs_label = juncs_score.argmax(dim=1)
            junction_features = loi_features[0].flatten(start_dim=1)[:,flat_index].T

        if self.require_valid_junctions:
            keep_mask = juncs_label > 0
            juncs_pred = juncs_pred[keep_mask]
            juncs_score = juncs_score[keep_mask]
            juncs_label = juncs_label[keep_mask]
            flat_index = flat_index[keep_mask]

        extra_info['time_proposal'] = time.time() - extra_info['time_proposal']
        extra_info['time_matching'] = time.time()
        if juncs_pred.size(0) > 1:
            dis_junc_to_end1, idx_junc_to_end1 = torch.sum((lines_pred[:,:2]-juncs_pred[:,None])**2,dim=-1).min(0)
            dis_junc_to_end2, idx_junc_to_end2 = torch.sum((lines_pred[:,2:] - juncs_pred[:, None]) ** 2, dim=-1).min(0)

            idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
            idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)

            # iskeep = (idx_junc_to_end_min < idx_junc_to_end_max)# * (dis_junc_to_end1< 10*10)*(dis_junc_to_end2<10*10)  # *(dis_junc_to_end2<100)
            iskeep = (idx_junc_to_end_min < idx_junc_to_end_max)*(dis_junc_to_end1< self.max_distance**2)*(dis_junc_to_end2<self.max_distance**2)
        else:
            iskeep = torch.zeros(1, dtype=torch.bool)

        some_lines_valid = iskeep.count_nonzero() > 0
        if some_lines_valid:
            idx_lines_for_junctions = torch.unique(
                torch.cat((idx_junc_to_end_min[iskeep,None],idx_junc_to_end_max[iskeep,None]),dim=1),
                dim=0)

            lines_adjusted = torch.cat((juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]),dim=1)

            extra_info['time_matching'] = time.time() - extra_info['time_matching']

            pooled_line_features = self.pooling(loi_features[0],lines_adjusted)

            # Filter lines
            line_logits = self.fc2(pooled_line_features)

            scores = line_logits.softmax(1)
            # TODO: Why is this done? And why not also filter the junctions?
            lines_score_valid = 1-scores[:,0]
            valid_mask = lines_score_valid > 0.05
            lines_final = lines_adjusted[valid_mask]
            pooled_line_features = pooled_line_features[valid_mask]
            line_logits = line_logits[valid_mask]


            # TODO: Supply edges for the junctions?
            unique_j_idx, l2j_idx = idx_lines_for_junctions[valid_mask].unique(return_inverse=True)
            juncs_final = juncs_pred[unique_j_idx]
            junction_features = junction_features[unique_j_idx]
            juncs_logits = juncs_logits[unique_j_idx]


            line_logits, juncs_logits = self.gnn_head(pooled_line_features, junction_features, l2j_idx, juncs_logits, line_logits)

            extra_info['time_verification'] = time.time()
            scores = line_logits.softmax(1)
            lines_score_valid = 1-scores[:,0]
            lines_label = scores.argmax(1)
            lines_score_label = torch.gather(scores, 1, lines_label.unsqueeze(1)).squeeze(1)

            juncs_score = juncs_logits.softmax(1)
            juncs_label = juncs_score.argmax(1)
            juncs_valid_score = 1-juncs_score[:,0]
            juncs_label_score = torch.gather(juncs_score, 1, juncs_label.unsqueeze(1)).squeeze(1)


            extra_info['time_verification'] = time.time() - extra_info['time_verification']
        else:
            extra_info['time_matching'] = time.time() - extra_info['time_matching']
            extra_info['time_verification'] = 0

        if annotations:
            width = annotations[0]['width']
            height = annotations[0]['height']
        else:
            width = images.size(3)
            height = images.size(2)

        sx = width/jloc_pred.size(3)
        sy = height/jloc_pred.size(2)

        juncs_pred[:,0] *= sx
        juncs_pred[:,1] *= sy
        extra_info['junc_prior_ver'] = juncs_pred
        lines_pred[:,0] *= sx
        lines_pred[:,1] *= sy
        lines_pred[:,2] *= sx
        lines_pred[:,3] *= sy
        extra_info['lines_prior_ver'] = lines_pred
        if some_lines_valid:
            lines_adjusted[:,0] *= sx
            lines_adjusted[:,1] *= sy
            lines_adjusted[:,2] *= sx
            lines_adjusted[:,3] *= sy
            extra_info['lines_prior_scoring'] = lines_adjusted
        else:
            extra_info['lines_prior_scoring'] = None


        if some_lines_valid and output_features:
            extra_info['gnn_line_features'] = pooled_line_features
            extra_info['gnn_junction_features'] = junction_features
            # extra_info['gnn_line2junc_idx'] = idx_lines_for_junctions
        else:
            extra_info['gnn_line_features'] = None
            extra_info['gnn_junction_features'] = None
            # extra_info['gnn_line2junc_idx'] = None

        output = {
            'num_proposals': 0,
            'filename': annotations[0]['filename'] if annotations else None,
            'width': width,
            'height': height,
        }

        if some_lines_valid:
            lines_final[:,0] *= sx
            lines_final[:,1] *= sy
            lines_final[:,2] *= sx
            lines_final[:,3] *= sy

            juncs_final[:,0] *= sx
            juncs_final[:,1] *= sy

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
                'num_proposals': lines_adjusted.size(0),
            })
        else:
            output.update({
                'lines_pred': torch.tensor([]),
                'juncs_pred': torch.tensor([])
                })
        return output, extra_info

    def forward_train(self, images, annotations = None):
        device = images.device

        # TODO: Caching the encoding and implement transforms for it might speed up training
        targets , metas = self.hafm_encoder(annotations)

        self.train_step += 1

        outputs, features = self.backbone(images)

        loss_dict = {
            'loss_md': torch.zeros(1, device=device),
            'loss_dis': torch.zeros(1, device=device),
            'loss_res': torch.zeros(1, device=device),
            'loss_jlabel': torch.zeros(1, device=device),
            'loss_joff': torch.zeros(1, device=device),
            'loss_pos': torch.zeros(1, device=device),
            'loss_neg': torch.zeros(1, device=device),
            'loss_cons': torch.zeros(1, device=device),
            'loss_gnn_jlabel': torch.zeros(1, device=device),
            'loss_gnn_pos': torch.zeros(1, device=device),
            'loss_gnn_neg': torch.zeros(1, device=device),
        }


        mask = targets['mask']
        if targets is not None:
            # Sample junction loss to include equal amounts of true and negative junctions.

            # b_sample_idx, e_sample_idx = [],[]
            # jlabel_target_flat = targets['jlabel'].flatten(start_dim=1)
            # for b_idx, batch_jlabel_flat in enumerate(jlabel_target_flat):
            #     true_idx = torch.nonzero(batch_jlabel_flat>0).flatten().tolist()
            #     false_idx = random.sample(set(range(batch_jlabel_flat.numel())) - set(true_idx), len(true_idx))
            #     e_sample_idx += true_idx + false_idx
            #     b_sample_idx += [b_idx]*2*len(true_idx)

            for nstack, output in enumerate(outputs):
                md_out, dis_out, res_out, jlabel_out, joff_out = self._get_output_dist(output)
                loss_map = torch.mean(F.l1_loss(md_out.sigmoid(), targets['md'],reduction='none'),dim=1,keepdim=True)
                loss_dict['loss_md']  += torch.mean(loss_map*mask) / torch.mean(mask)
                loss_map = F.l1_loss(dis_out.sigmoid(), targets['dis'], reduction='none')
                loss_dict['loss_dis'] += torch.mean(loss_map*mask) /torch.mean(mask)
                loss_residual_map = F.l1_loss(res_out.sigmoid(), loss_map, reduction='none')
                loss_dict['loss_res'] += torch.mean(loss_residual_map*mask)/torch.mean(mask)
                #TODO: Correct dimensions?
                loss_dict['loss_jlabel'] += self.junction_label_loss(
                    jlabel_out.flatten(start_dim=2),
                    targets['jlabel'].flatten(start_dim=1),
                    # jlabel_out.flatten(start_dim=2)[b_sample_idx, :, e_sample_idx],
                    # jlabel_target_flat[b_sample_idx, e_sample_idx]
                    )
                loss_dict['loss_joff'] += sigmoid_l1_loss(joff_out, targets['joff'], -0.5, targets['jloc'])

        loi_features = self.fc1(features)
        md_out, dis_out, res_out, jlabel_out, joff_out = self._get_output_dist(outputs[0])
        md_pred = md_out.sigmoid()
        dis_pred = dis_out.sigmoid()
        res_pred = res_out.sigmoid()
        jlabel_prob = jlabel_out.softmax(1)
        jlabel = jlabel_prob.argmax(1)
        jloc_pred = 1-jlabel_prob[:,0,None]
        joff_pred= joff_out.sigmoid() - 0.5

        lines_batch = []
        extra_info = {
        }

        batch_size = md_pred.size(0)

        for i, (md_pred_per_im, dis_pred_per_im,res_pred_per_im,meta) in enumerate(zip(md_pred, dis_pred,res_pred,metas)):
            lines_pred = []
            if self.use_residual:
                for scale in [-1.0,0.0,1.0]:
                    _ = self.proposal_lines(md_pred_per_im, dis_pred_per_im+scale*res_pred_per_im).view(-1, 4)
                    lines_pred.append(_)
            else:
                lines_pred.append(self.proposal_lines(md_pred_per_im, dis_pred_per_im).view(-1, 4))
            lines_pred = torch.cat(lines_pred)
            junction_gt = meta['junc']
            N = junction_gt.size(0)

            juncs_pred, juncs_valid_score, flat_index = get_junctions(non_maximum_suppression(jloc_pred[i]),joff_pred[i], topk=min(N*2+2,self.n_dyn_junc))
            junction_features = loi_features[i].flatten(start_dim=1)[:,flat_index].T
            # print('jlabel_out',jlabel_out.shape)
            # print('junction_features',junction_features.shape)
            juncs_logits = (jlabel_out.flatten(start_dim=2)[i,:,flat_index]).T
            # jtargets = targets['jlabel'][i].flatten()[flat_index]
            # print('juncs_logits1',juncs_logits.shape)
            # print('jtargets',jtargets.shape)
            # print('targets[jlabel][i]', targets['jlabel'][i].size())


            if self.require_valid_junctions:
                keep_mask = jlabel.flatten()[flat_index] > 0
                juncs_pred = juncs_pred[keep_mask]
                juncs_valid_score = juncs_valid_score[keep_mask]
                juncs_logits = juncs_logits[keep_mask]

            # No junctions, just add static training examples
            if juncs_pred.size(0) < 2:
                logits = self.pooling(loi_features[i],meta['lpre'])
                loss_ = self.loss(logits, meta['lpre_label'])

                loss_positive = loss_[meta['lpre_label']>0].mean()
                loss_negative = loss_[meta['lpre_label']==0].mean()

                loss_dict['loss_pos'] += loss_positive/batch_size
                loss_dict['loss_neg'] += loss_negative/batch_size
                continue

            dis_junc_to_end1, idx_junc_to_end1 = torch.sum((lines_pred[:,:2]-juncs_pred[:,None])**2,dim=-1).min(0)
            dis_junc_to_end2, idx_junc_to_end2 = torch.sum((lines_pred[:, 2:] - juncs_pred[:, None]) ** 2, dim=-1).min(0)

            idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
            idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)
            iskeep = idx_junc_to_end_min<idx_junc_to_end_max
            idx_lines_for_junctions = torch.cat((idx_junc_to_end_min[iskeep,None],idx_junc_to_end_max[iskeep,None]),dim=1).unique(dim=0)
            # idx_lines_for_junctions_mirror = torch.cat((idx_lines_for_junctions[:,1,None],idx_lines_for_junctions[:,0,None]),dim=1)
            # idx_lines_for_junctions = torch.cat((idx_lines_for_junctions, idx_lines_for_junctions_mirror))
            lines_adjusted = torch.cat((juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]),dim=1)

            cost_, match_ = torch.sum((juncs_pred-junction_gt[:,None])**2,dim=-1).min(0)
            match_[cost_>1.5*1.5] = N
            Lpos = meta['Lpos']
            labels = Lpos[match_[idx_lines_for_junctions[:,0]],match_[idx_lines_for_junctions[:,1]]]

            iskeep = torch.zeros_like(labels, dtype= torch.bool)
            cdx = labels.nonzero().flatten()

            if len(cdx) > self.n_dyn_posl:
                perm = torch.randperm(len(cdx),device=device)[:self.n_dyn_posl]
                cdx = cdx[perm]

            iskeep[cdx] = 1

            if self.n_dyn_othr2 >0 :
                cdx = (labels==0).nonzero().flatten()
                if len(cdx) > self.n_dyn_othr2:
                    perm = torch.randperm(len(cdx), device=device)[:self.n_dyn_othr2]
                    cdx = cdx[perm]
                iskeep[cdx] = 1

            # print('targets',targets['jlabel'].shape)
            #
            # print('junction_features',junction_features.shape)
            # print('idx_lines_for_junctions',idx_lines_for_junctions.shape)
            all_lines = torch.cat((lines_adjusted,meta['lpre']))
            all_labels = torch.cat((labels,meta['lpre_label']))
            # print('all_lines',all_lines.shape)
            # print('all_labels',all_labels.shape)
            pooled_line_features = self.pooling(loi_features[i],all_lines)
            line_logits_no_gnn = self.fc2(pooled_line_features)
            # print('pooled_line_features',pooled_line_features.shape)
            line_logits, juncs_logits = self.gnn_head(pooled_line_features, junction_features, idx_lines_for_junctions, juncs_logits, line_logits_no_gnn)
            all_iskeep = torch.cat((iskeep,torch.ones_like(meta['lpre_label'], dtype= torch.bool)))


            # labels_selected = labels[iskeep]
            #
            # lines_for_train = torch.cat((lines_selected,meta['lpre']))
            # labels_for_train = torch.cat((labels_selected,meta['lpre_label']))

            # print('targets',targets['jlabel'].shape)
            # pooled_line_features = self.pooling(loi_features[i],lines_for_train)
            # print('pooled_line_features',pooled_line_features.shape)
            # print('junction_features',junction_features.shape)
            # print('idx_lines_for_junctions',idx_lines_for_junctions.shape)

            # line_logits, juncs_logits = self.gnn_head(pooled_line_features, junction_features, idx_lines_for_junctions, juncs_logits)
            # print('line_logits',line_logits.shape)
            # print('juncs_logits2',juncs_logits.shape)
            # print('jlabel',targets['jlabel'].shape)
            # print('jlabel',targets['jlabel'][i].flatten()[flat_index].shape)
            jtargets = targets['jlabel'][i].flatten()[flat_index]
            jidx_keep = idx_lines_for_junctions[iskeep].flatten().unique()
            # print('jidx_keep',jidx_keep.size())
            # print('jtargets',jidx_keep.size())
            # print('juncs_logits',juncs_logits.size())
            # print('idx_lines_for_junctions[iskeep]',idx_lines_for_junctions[iskeep].size())
            loss_dict['loss_gnn_jlabel'] += self.gnn_junction_label_loss(
                juncs_logits[jidx_keep],
                jtargets[jidx_keep]
                )

            selected_logits = line_logits[all_iskeep]
            selected_labels = all_labels[all_iskeep]
            # print('selected_logits',selected_logits.shape)
            # print('selected_labels',selected_labels.shape)
            loss_gnn = self.loss(selected_logits, selected_labels)

            loss_dict['loss_gnn_pos'] +=  loss_gnn[selected_labels>0].mean()/batch_size
            loss_dict['loss_gnn_neg'] += loss_gnn[selected_labels==0].mean()/batch_size

            selected_logits_nognn = line_logits_no_gnn[all_iskeep]
            loss_no_gnn = self.loss(selected_logits_nognn, selected_labels)

            loss_dict['loss_pos'] += loss_no_gnn[selected_labels>0].mean()/batch_size
            loss_dict['loss_neg'] += loss_no_gnn[selected_labels==0].mean()/batch_size


            # Penalize if a line classifies as valid, but the junctions does not.
            lines_selected = lines_adjusted[iskeep]
            idx_lines_for_junctions = idx_lines_for_junctions[iskeep]
            log_prob_valid = 1-line_logits[:lines_selected.size(0)].softmax(1)[:,0]
            loss_cons = log_prob_valid.unsqueeze(1)*(1-juncs_valid_score[idx_lines_for_junctions])
            #TODO: Would be better to take negative log likelihood on loss_cons since the loss now is bounded between 0 and 1.

            loss_dict['loss_cons'] += loss_cons.mean()/batch_size

        return loss_dict, extra_info

    def proposal_lines(self, md_maps, dis_maps, scale=5.0):
        """

        :param md_maps: 3xhxw, the range should be (0,1) for every element
        :param dis_maps: 1xhxw
        :return:
        """
        device = md_maps.device
        height, width = md_maps.size(1), md_maps.size(2)
        _y = torch.arange(0,height,device=device).float()
        _x = torch.arange(0,width, device=device).float()

        y0,x0 = torch.meshgrid(_y,_x)
        md_ = (md_maps[0]-0.5)*np.pi*2
        st_ = md_maps[1]*np.pi/2
        ed_ = -md_maps[2]*np.pi/2

        cs_md = torch.cos(md_)
        ss_md = torch.sin(md_)

        cs_st = torch.cos(st_).clamp(min=1e-3)
        ss_st = torch.sin(st_).clamp(min=1e-3)

        cs_ed = torch.cos(ed_).clamp(min=1e-3)
        ss_ed = torch.sin(ed_).clamp(max=-1e-3)

        x_standard = torch.ones_like(cs_st)

        y_st = ss_st/cs_st
        y_ed = ss_ed/cs_ed

        x_st_rotated =  (cs_md - ss_md*y_st)*dis_maps[0]*scale
        y_st_rotated =  (ss_md + cs_md*y_st)*dis_maps[0]*scale

        x_ed_rotated =  (cs_md - ss_md*y_ed)*dis_maps[0]*scale
        y_ed_rotated = (ss_md + cs_md*y_ed)*dis_maps[0]*scale

        x_st_final = (x_st_rotated + x0).clamp(min=0,max=width-1)
        y_st_final = (y_st_rotated + y0).clamp(min=0,max=height-1)

        x_ed_final = (x_ed_rotated + x0).clamp(min=0,max=width-1)
        y_ed_final = (y_ed_rotated + y0).clamp(min=0,max=height-1)

        lines = torch.stack((x_st_final,y_st_final,x_ed_final,y_ed_final)).permute((1,2,0))

        return  lines#, normals

    def proposal_lines_new(self, md_maps, dis_maps, residual_maps, scale=5.0):
        """

        :param md_maps: 3xhxw, the range should be (0,1) for every element
        :param dis_maps: 1xhxw
        :return:
        """
        device = md_maps.device
        sign_pad     = torch.tensor([-1,0,1],device=device,dtype=torch.float32).reshape(3,1,1)

        if residual_maps is None:
            dis_maps_new = dis_maps.repeat((1,1,1))
        else:
            dis_maps_new = dis_maps.repeat((3,1,1))+sign_pad*residual_maps.repeat((3,1,1))
        height, width = md_maps.size(1), md_maps.size(2)
        _y = torch.arange(0,height,device=device).float()
        _x = torch.arange(0,width, device=device).float()

        y0,x0 = torch.meshgrid(_y,_x)
        md_ = (md_maps[0]-0.5)*np.pi*2
        st_ = md_maps[1]*np.pi/2
        ed_ = -md_maps[2]*np.pi/2

        cs_md = torch.cos(md_)
        ss_md = torch.sin(md_)

        cs_st = torch.cos(st_).clamp(min=1e-3)
        ss_st = torch.sin(st_).clamp(min=1e-3)

        cs_ed = torch.cos(ed_).clamp(min=1e-3)
        ss_ed = torch.sin(ed_).clamp(max=-1e-3)

        y_st = ss_st/cs_st
        y_ed = ss_ed/cs_ed

        x_st_rotated = (cs_md-ss_md*y_st)[None]*dis_maps_new*scale
        y_st_rotated =  (ss_md + cs_md*y_st)[None]*dis_maps_new*scale

        x_ed_rotated =  (cs_md - ss_md*y_ed)[None]*dis_maps_new*scale
        y_ed_rotated = (ss_md + cs_md*y_ed)[None]*dis_maps_new*scale

        x_st_final = (x_st_rotated + x0[None]).clamp(min=0,max=width-1)
        y_st_final = (y_st_rotated + y0[None]).clamp(min=0,max=height-1)

        x_ed_final = (x_ed_rotated + x0[None]).clamp(min=0,max=width-1)
        y_ed_final = (y_ed_rotated + y0[None]).clamp(min=0,max=height-1)

        lines = torch.stack((x_st_final,y_st_final,x_ed_final,y_ed_final)).permute((1,2,3,0))

        # normals = torch.stack((cs_md,ss_md)).permute((1,2,0))

        return  lines#, normals
