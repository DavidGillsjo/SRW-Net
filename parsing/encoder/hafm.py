import torch
import numpy as np
from torch.utils.data.dataloader import default_collate

from parsing import _C

class HAFMencoder(object):
    def __init__(self, cfg):
        self.dis_th = cfg.ENCODER.DIS_TH
        self.ang_th = cfg.ENCODER.ANG_TH
        self.num_static_pos_lines = cfg.ENCODER.NUM_STATIC_POS_LINES
        self.num_static_neg_lines = cfg.ENCODER.NUM_STATIC_NEG_LINES
    def __call__(self,annotations):
        targets = []
        metas   = []
        for ann in annotations:
            t,m = self._process_per_image(ann)
            targets.append(t)
            metas.append(m)

        return default_collate(targets),metas

    def adjacent_matrix(self, n, edges, device, labels = None):
        dtype = torch.bool if labels is None else torch.long
        mat = torch.zeros(n+1,n+1,dtype=dtype,device=device)
        if edges.size(0)>0:
            val = True if labels is None else labels
            mat[edges[:,0], edges[:,1]] = val
            mat[edges[:,1], edges[:,0]] = val

        return mat

    def _process_per_image(self,ann):
        junctions = ann['junctions']
        device = junctions.device
        height, width = ann['height'], ann['width']
        jmap = torch.zeros((height,width),device=device)
        joff = torch.zeros((2,height,width),device=device,dtype=torch.float32)

        xint,yint = junctions[:,0].long(), junctions[:,1].long()
        off_x = junctions[:,0] - xint.float()-0.5
        off_y = junctions[:,1] - yint.float()-0.5
        joff[0,yint,xint] = off_x
        joff[1,yint,xint] = off_y
        jmap[yint,xint] = 1

        jlabel = torch.zeros((height,width), device=device, dtype=torch.long)
        if 'junctions_semantic' in ann:
            jlabel[yint, xint] = ann['junctions_semantic']
        else:
            joccluded = ann['junc_occluded']
            jlabel[yint[joccluded],xint[joccluded]] = 1
            jlabel[yint[~joccluded],xint[~joccluded]] = 2


        edges_positive = ann['edges_positive']
        lpos_labels = ann['edges_semantic']
        edges_negative = ann['edges_negative']

        pos_mat = self.adjacent_matrix(junctions.size(0),edges_positive,device,labels=lpos_labels)
        lines = torch.cat((junctions[edges_positive[:,0]], junctions[edges_positive[:,1]]),dim=-1)
        lines_neg = torch.cat((junctions[edges_negative[:2000,0]],junctions[edges_negative[:2000,1]]),dim=-1)
        lmap, _, _ = _C.encodels(lines,height,width,height,width,lines.size(0))

        pos_permutation = np.random.permutation(np.arange(lpos_labels.shape[0]))[:self.num_static_pos_lines]
        lpos = lines[pos_permutation].to(device)
        lpos_labels = lpos_labels[pos_permutation].to(device)
        lneg = np.random.permutation(lines_neg.cpu().numpy())[:self.num_static_neg_lines]
        lneg = torch.from_numpy(lneg).to(device)

        lpre = torch.cat((lpos,lneg),dim=0)
        _swap = (torch.rand(lpre.size(0))>0.5).to(device)
        lpre[_swap] = lpre[_swap][:,[2,3,0,1]]
        # Constuct labels for training line classification
        lpre_label = torch.cat(
            [
                lpos_labels.to(device),
                torch.zeros(lneg.size(0),device=device, dtype=torch.long)
             ])

        meta = {
            'junc': junctions,
            'Lpos':   pos_mat,
            'lpre':      lpre,
            'lpre_label': lpre_label,
            'lines':     lines,
        }


        dismap = torch.sqrt(lmap[0]**2+lmap[1]**2)[None]
        def _normalize(inp):
            mag = torch.sqrt(inp[0]*inp[0]+inp[1]*inp[1])
            return inp/(mag+1e-6)
        md_map = _normalize(lmap[:2])
        st_map = _normalize(lmap[2:4])
        ed_map = _normalize(lmap[4:])

        md_ = md_map.reshape(2,-1).t()
        st_ = st_map.reshape(2,-1).t()
        ed_ = ed_map.reshape(2,-1).t()
        Rt = torch.cat(
                (torch.cat((md_[:,None,None,0],md_[:,None,None,1]),dim=2),
                 torch.cat((-md_[:,None,None,1], md_[:,None,None,0]),dim=2)),dim=1)
        R = torch.cat(
                (torch.cat((md_[:,None,None,0], -md_[:,None,None,1]),dim=2),
                 torch.cat((md_[:,None,None,1], md_[:,None,None,0]),dim=2)),dim=1)

        Rtst_ = torch.matmul(Rt, st_[:,:,None]).squeeze(-1).t()
        Rted_ = torch.matmul(Rt, ed_[:,:,None]).squeeze(-1).t()
        swap_mask = (Rtst_[1]<0)*(Rted_[1]>0)
        pos_ = Rtst_.clone()
        neg_ = Rted_.clone()
        temp = pos_[:,swap_mask]
        pos_[:,swap_mask] = neg_[:,swap_mask]
        neg_[:,swap_mask] = temp

        pos_[0] = pos_[0].clamp(min=1e-9)
        pos_[1] = pos_[1].clamp(min=1e-9)
        neg_[0] = neg_[0].clamp(min=1e-9)
        neg_[1] = neg_[1].clamp(max=-1e-9)

        mask = ((pos_[1]>self.ang_th)*(neg_[1]<-self.ang_th)*(dismap.view(-1)<=self.dis_th)).float()

        pos_map = pos_.reshape(-1,height,width)
        neg_map = neg_.reshape(-1,height,width)

        md_angle  = torch.atan2(md_map[1], md_map[0])
        pos_angle = torch.atan2(pos_map[1],pos_map[0])
        neg_angle = torch.atan2(neg_map[1],neg_map[0])

        pos_angle_n = pos_angle/(np.pi/2)
        neg_angle_n = -neg_angle/(np.pi/2)
        md_angle_n  = md_angle/(np.pi*2) + 0.5
        mask    = mask.reshape(height,width)

        hafm_ang = torch.cat((md_angle_n[None],pos_angle_n[None],neg_angle_n[None],),dim=0)
        hafm_dis   = dismap.clamp(max=self.dis_th)/self.dis_th
        mask = mask[None]
        target = {'jloc':jmap[None],
                'jlabel':jlabel[None],
                'joff':joff,
                'md': hafm_ang,
                'dis': hafm_dis,
                'mask': mask
               }
        return target, meta
