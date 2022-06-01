import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import linear_sum_assignment

"""
Evaluation according to https://github.com/leVirve/lsun-room/blob/master/lib/lsun_toolkit/cornerError.m
distmat = pdist2( prediction, groundtruth);
[Matching,Cost] = Hungarian(distmat);

ptCost = sum(Cost./norm(sz)) + abs(size(prediction,1)-size(groundtruth,1))*1/3;
ptCost = ptCost./max( size(prediction,1), size(groundtruth,1));

"""
def LSUN_keypoint_error(junctions, edges, score, junction_gt, thresholds, resolution = [128,128]):
    junctions_scores = np.zeros(junctions.shape[0])
    for idx, e in enumerate(edges):
        junctions_scores[e] = np.maximum(score[idx], junctions_scores[e])

    min_mask = junctions_scores >= thresholds[0]
    junctions_scores = junctions_scores[min_mask]
    junctions = junctions[min_mask]
    dmat = cdist(junctions, junction_gt, metric='euclidean')
    dmat /= np.linalg.norm(resolution)

    nbr_gt = junction_gt.shape[0]

    kp_error = np.zeros_like(thresholds)
    for idx, t in enumerate(thresholds):
        t_mask = junctions_scores >= t
        masked_junctions = junctions[t_mask]
        masked_dmat = dmat[t_mask]
        nbr_det = masked_junctions.shape[0]
        row_ind, col_ind = linear_sum_assignment(masked_dmat)

        error_sum = masked_dmat[row_ind, col_ind].sum() + np.abs(nbr_gt - nbr_det)*1/3
        kp_error[idx] = error_sum/np.maximum(nbr_gt, nbr_det)

    return thresholds, kp_error



"""
The structural AP is defined in terms of squared distances, so we make AP for junctions the same.
"""

def TPFP_labels(lines_dt, lines_gt, labels, labels_gt, threshold):
    tp = np.zeros(labels.shape, dtype=np.bool)
    fp = np.zeros_like(tp)
    for l in  np.unique(labels):
        gt_mask = (l==labels_gt)
        dt_mask = (l==labels)
        if np.any(gt_mask):
            tp[dt_mask], fp[dt_mask] = TPFP(lines_dt[dt_mask],
                                            lines_gt[gt_mask],
                                            threshold)
        else:
            tp[dt_mask] = False
            fp[dt_mask] = True

    return tp, fp


def _diff_to_tpfp(diff, threshold):
    choice = np.argmin(diff,axis=1)
    dist = diff[np.arange(diff.shape[0]),choice]

    hit = np.zeros(diff.shape[1], np.bool)
    tp = np.zeros(diff.shape[0], np.bool)
    fp = np.zeros(diff.shape[0],np.bool)

    for i in range(diff.shape[0]):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = True
        else:
            fp[i] = True


    return tp, fp

def TPFP(lines_dt, lines_gt, threshold):
    lines_dt = lines_dt.reshape(-1,2,2)[:,:,::-1]
    lines_gt = lines_gt.reshape(-1,2,2)[:,:,::-1]
    diff = ((lines_dt[:, None, :, None] - lines_gt[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )

    tp, fp = _diff_to_tpfp(diff, threshold)

    return tp, fp


def TPFP_junctions(junctions, junctions_gt, threshold):
    diff = np.sum((junctions[:,None] - junctions_gt[None,:])**2, axis=-1)
    tp, fp = _diff_to_tpfp(diff, threshold)
    return tp, fp

def TPFP_junctions_labels(junctions, junctions_gt, labels, labels_gt, threshold):
    tp = np.zeros(labels.shape, dtype=np.bool)
    fp = np.zeros_like(tp)
    for l in  np.unique(labels):
        gt_mask = (l==labels_gt)
        dt_mask = (l==labels)
        if np.any(gt_mask):
            tp[dt_mask], fp[dt_mask] = TPFP_junctions(junctions[dt_mask],
                                                      junctions_gt[gt_mask],
                                                      threshold)
        else:
            tp[dt_mask] = False
            fp[dt_mask] = True

    return tp, fp


def PR(tp, fp, scores, n_gt):
    idx = np.argsort(scores)[::-1]
    tp = np.cumsum(tp[idx])
    fp = np.cumsum(fp[idx])
    rc = tp/n_gt
    pc = tp/np.maximum(tp+fp,1e-9)
    return pc, rc

def AP(precision, recall):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]

    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])

    return ap*100

# assume one class is the background class.
class sAPmeter:
    def __init__(self, labels = [], threshold = 10, bkg_idx = 0):
        self.tp_dict, self.fp_dict, self.scores_dict, self.n_gt_dict = {},{},{},{}
        self.labels_idx = [idx for idx in range(len(labels)) if idx  != bkg_idx]
        self.labels = [labels[idx] for idx in self.labels_idx]
        self.threshold = threshold
        self.tpfp_func = TPFP
        for l in list(self.labels) + ['all']:
            self.tp_dict[l] = []
            self.fp_dict[l] = []
            self.scores_dict[l] = []
            self.n_gt_dict[l] = 0

        self.update = self._update_with_label if self.labels else self._update_no_label

    def _update_no_label(self,lines_pred, lines_gt, scores):
        tp, fp = self.tpfp_func(lines_pred, lines_gt, self.threshold)
        self.n_gt_dict['all'] += lines_gt.shape[0]
        self.tp_dict['all'].append(tp)
        self.fp_dict['all'].append(fp)
        self.scores_dict['all'].append(scores)

    # Assume idx 0 is a background class
    def _update_with_label(self, lines_pred, lines_gt, scores, labels_pred, labels_gt):
        for l, l_name in zip(self.labels_idx, self.labels):
            gt_mask = (l==labels_gt)
            nbr_gt = np.count_nonzero(gt_mask)
            dt_mask = (l==labels_pred)
            if np.any(gt_mask):
                tp, fp = self.tpfp_func(lines_pred[dt_mask],
                                        lines_gt[gt_mask],
                                        self.threshold)
            else:
                nbr_det = np.count_nonzero(dt_mask)
                tp = np.zeros(nbr_det, dtype=np.bool)
                fp = np.ones(nbr_det, dtype=np.bool)

            for l_update in [l_name, 'all']:
                self.tp_dict[l_update].append(tp)
                self.fp_dict[l_update].append(fp)
                self.scores_dict[l_update].append(scores[dt_mask])
                self.n_gt_dict[l_update] += nbr_gt

    def evaluate(self):
        pcs, rcs, sAP = {},{},{}
        for k in self.tp_dict:
            tp_list = self.tp_dict[k]
            fp_list = self.fp_dict[k]
            n_gt = self.n_gt_dict[k]
            scores_list = self.scores_dict[k]
            if len(tp_list) == 0 or n_gt == 0:
                rcs[k], pcs[k], sAP[k] = np.zeros(1), np.zeros(1), 0
                continue
            tp_list = np.concatenate(tp_list)
            fp_list = np.concatenate(fp_list)
            scores_list = np.concatenate(scores_list)
            pcs[k], rcs[k] = PR(tp_list, fp_list, scores_list, n_gt)
            sAP[k] = AP(pcs[k],rcs[k])

        if self.labels:
            mAP = 0
            for l in self.labels:
                mAP += sAP[l]
            sAP['mean'] = mAP / len(self.labels)


        return rcs, pcs, sAP

class jAPmeter(sAPmeter):
    def __init__(self, labels = [], threshold = 10):
        super().__init__(labels, threshold)
        self.tpfp_func = TPFP_junctions


def evalulate_sap(result_list, annotations_dict, thresholds, labels):
    eval_meters = {}
    for et in ['valid', 'label']:
        meter_labels = labels if et == 'label' else []
        eval_meters[et] = {t:sAPmeter(labels=meter_labels, threshold=t) for t in thresholds}

    for res in result_list:
        filename = res['filename']
        gt = annotations_dict[filename]
        if len(res.get('lines_pred',[])) == 0:
            continue
        lines_pred = np.array(res['lines_pred'],dtype=np.float32)

        # import pdb; pdb.set_trace()
        lines_pred[:,0] *= 128/float(res['width'])
        lines_pred[:,1] *= 128/float(res['height'])
        lines_pred[:,2] *= 128/float(res['width'])
        lines_pred[:,3] *= 128/float(res['height'])

        junctions_gt = np.array(gt['junctions'], dtype=np.float32)
        lines_gt = []
        for e in gt['edges_positive']:
            lines_gt.append(
               junctions_gt[e[0]].tolist() + junctions_gt[e[1]].tolist()
            )
        lines_gt = np.array(lines_gt, dtype=np.float32)
        lines_gt[:,0]  *= 128/float(gt['width'])
        lines_gt[:,1]  *= 128/float(gt['height'])
        lines_gt[:,2]  *= 128/float(gt['width'])
        lines_gt[:,3]  *= 128/float(gt['height'])

        labels_gt = np.array(gt['edges_semantic'], dtype=np.int)

        for k, eval_type_meters in eval_meters.items():
            scores = np.array(res['lines_{}_score'.format(k)],dtype=np.float32)

            if k == 'label':
                for _, meter in eval_type_meters.items():
                    labels = np.array(res['lines_label'], dtype=np.float32)
                    meter.update(lines_pred, lines_gt, scores, labels, labels_gt)
            else:
                for _, meter in eval_type_meters.items():
                    meter.update(lines_pred, lines_gt, scores)

    rcs, pcs, sAP = {},{},{}

    for eval_type, eval_type_meters in eval_meters.items():
        rcs[eval_type] = e_rcs = {}
        pcs[eval_type] = e_pcs = {}
        sAP[eval_type] = e_sAP = {}
        msAP_acc = {}
        for t, meter in eval_type_meters.items():
            e_rcs[t], e_pcs[t], e_sAP[t] = meter.evaluate()
            for ap_type, ap in e_sAP[t].items():
                if ap_type in msAP_acc:
                    msAP_acc[ap_type].append(ap)
                else:
                    msAP_acc[ap_type] = [ap]
        e_sAP['mean'] = {ap_type: float(np.mean(ap_list)) for ap_type, ap_list in msAP_acc.items()}


    return rcs, pcs, sAP


def evalulate_jap(result_list, annotations_dict, thresholds, labels):
    eval_meters = {}
    for et in ['valid', 'label', 'label_line_valid']:
        meter_labels = labels if et.startswith('label') else []
        eval_meters[et] = {t:jAPmeter(labels=meter_labels, threshold=t) for t in thresholds}

    for res in result_list:
        filename = res['filename']
        gt = annotations_dict[filename]
        if len(res.get('juncs_pred',[])) == 0:
            continue

        juncs_pred = np.array(res['juncs_pred'],dtype=np.float32)
        juncs_pred[:,0] *= 128/float(res['width'])
        juncs_pred[:,1] *= 128/float(res['height'])

        juncs_gt = np.array(gt['junctions'],dtype=np.float32)
        juncs_gt[:,0] *= 128/float(res['width'])
        juncs_gt[:,1] *= 128/float(res['height'])

        if 'junctions_semantic' in gt:
            labels_gt = np.array(gt['junctions_semantic'],dtype=np.int)
        else:
            occluded = np.array(gt['junc_occluded'], dtype=np.bool)
            labels_gt = np.zeros(occluded.shape, dtype=np.int)
            labels_gt[occluded] = 1
            labels_gt[~occluded] = 2

        for k, eval_type_meters in eval_meters.items():

            if k == 'label':
                labels = np.array(res['juncs_label'], dtype=np.int)
                scores = np.array(res['juncs_label_score'])
                for _, meter in eval_type_meters.items():
                    meter.update(juncs_pred, juncs_gt, scores, labels, labels_gt)
            elif k == 'label_line_valid' and 'line2junc_idx' in res:
                scores = np.array(res['juncs_score'])
                line_labels = np.array(res['lines_label'], dtype=np.int)
                line2junc_idx = np.array(res['line2junc_idx'], dtype=np.int)
                #Find junctions which lines are valid
                valid_jidx = np.unique(line2junc_idx[line_labels > 0])
                scores = scores[valid_jidx]
                k_labels = np.argmax(scores[:,1:], axis=1) + 1
                scores = scores[np.arange(scores.shape[0]),k_labels]
                k_juncs_pred = juncs_pred[valid_jidx]
                for _, meter in eval_type_meters.items():
                    meter.update(k_juncs_pred, juncs_gt, scores, k_labels, labels_gt)
            elif k == 'valid':
                scores = np.array(res['juncs_valid_score'])
                for _, meter in eval_type_meters.items():
                    meter.update(juncs_pred, juncs_gt, scores)

    rcs, pcs, jAP = {},{},{}

    for eval_type, eval_type_meters in eval_meters.items():
        rcs[eval_type] = e_rcs = {}
        pcs[eval_type] = e_pcs = {}
        jAP[eval_type] = e_jAP = {}
        mjAP_acc = {}
        for t, meter in eval_type_meters.items():
            e_rcs[t], e_pcs[t], e_jAP[t] = meter.evaluate()
            for ap_type, ap in e_jAP[t].items():
                if ap_type in mjAP_acc:
                    mjAP_acc[ap_type].append(ap)
                else:
                    mjAP_acc[ap_type] = [ap]
        e_jAP['mean'] = {ap_type: float(np.mean(ap_list)) for ap_type, ap_list in mjAP_acc.items()}

    return rcs, pcs, jAP

def evalulate_lsun_kp(result_list, annotations_dict, thresholds = np.linspace(0.1,0.9, 9)):
    lsun_acc = []

    for res in result_list:
        filename = res['filename']
        gt = annotations_dict[filename]
        if len(res.get('lines_pred',[])) == 0:
            continue
        edges = np.array(res['line2junc_idx'],dtype=np.int)
        junctions = np.array(res['juncs_pred'],dtype=np.float32)
        scores = np.array(res['lines_valid_score'],dtype=np.float32)
        junction_gt = np.array(gt['junctions'],dtype=np.float32)

        t, e = LSUN_keypoint_error(junctions, edges, scores, junction_gt, thresholds, resolution = [res['width'],res['height']])
        lsun_acc.append(e)

    lsun_mean = np.vstack(lsun_acc).mean(axis=0) if lsun_acc else np.zeros_like(thresholds)

    return thresholds, lsun_mean


if __name__ == "__main__":

    nbr_edges = 5
    nbr_junctions = 2*nbr_edges
    junctions_gt = np.random.ranf([nbr_junctions,2])
    edges = np.reshape(np.arange(nbr_junctions), [nbr_edges,2])
    # scores = np.random.ranf(nbr_edges)
    scores = np.ones(nbr_edges)*0.5
    junctions_det = junctions_gt #+ np.random.normal(0, 0.1, [nbr_junctions,2])

    kp_thres , kp_error = LSUN_keypoint_error(junctions_det, edges, scores, junctions_gt, thresholds = [0, 0.5,0.8],resolution=[1,1])


    # random_lines = np.random.random([50, 4])*100
    # gt_lines = np.copy(random_lines[-25:])+np.random.randn(25,4)*0.8*np.linspace(0,1, 25)[::-1,None]
    # scores = np.linspace(0, 1, random_lines.shape[0])
    # labels = np.random.randint(1,5, [50])
    # labels_gt = labels[-25:] + np.random.randint(0,1, [25])
    #
    #
    # tp, fp = TPFP_labels(random_lines, gt_lines, 5,labels_gt,  labels)
    # # tp, fp = TPFP(random_lines, gt_lines, 1)
    #
    # pc, rc, = PR(tp, fp, scores, gt_lines.shape[0])
    # ap = AP(pc, rc)
    # print('AP', ap)
    # from matplotlib import pyplot as plt
    # import sys
    # plt.figure()
    # plt.plot(rc, pc, '.-')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.savefig('lines_PR.png')
    # plt.close()
    # sys.exit()
    #
    # random_junctions = np.random.random([30, 2])*100
    # gt_junctions = np.copy(random_junctions[-25:])+np.random.randn(25,2)*0.02*np.linspace(0,1, 25)[::-1,None]
    # # gt_junctions[0,1] += 0.01
    # scores = np.linspace(0, 1, random_junctions.shape[0])
    #
    # tp, fp = TPFP_junctions(random_junctions, gt_junctions, 0.01)
    # pc, rc, = PR(tp, fp, scores, gt_junctions.shape[0])
    # ap = AP(pc, rc)
    # plt.figure()
    # plt.plot(rc, pc, '.-')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.savefig('junctions_PR.png')
    # plt.close()
