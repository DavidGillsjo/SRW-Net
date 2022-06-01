import os
import os.path as osp

class DatasetCatalog(object):

    DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__),
                '..','..','data'))

    DATASETS = {
        'wireframe_train': {
            'img_dir': 'wireframe/images',
            'ann_file': 'wireframe/train.json',
        },
        'wireframe_test': {
            'img_dir': 'wireframe/images',
            'ann_file': 'wireframe/test.json',
        },
        'york_test': {
            'img_dir': 'york/images',
            'ann_file': 'york/test.json',
        }}
    DATASETS.update({
        'structured3D_{}'.format(d): {
            'img_dir': 'Structured3D_wf/images',
            'ann_file': 'Structured3D_wf/{}.json'.format(d)}
        for d in ('train', 'train_mini', 'test', 'test_mini', 'val', 'val_mini')
    })
    DATASETS.update({
        'structured3D_wfc_{}'.format(d): {
            'img_dir': 'Structured3D_wf_wfc/images',
            'ann_file': 'Structured3D_wf_wfc/{}.json'.format(d)}
        for d in ('train', 'train_mini', 'test', 'test_mini', 'val', 'val_mini')
    })
    DATASETS.update({
        'structured3D_rwd7_{}'.format(d): {
            'img_dir': 'Structured3D_wf_rwd_7label/images',
            'ann_file': 'Structured3D_wf_rwd_7label/{}.json'.format(d)}
        for d in ('train', 'train_mini', 'test', 'test_mini', 'val', 'val_mini')
    })
    DATASETS.update({
        'LSUN_{}'.format(d): {
            'img_dir': 'LSUN_wf/images',
            'ann_file': 'LSUN_wf/{}.json'.format(d)}
        for d in ('train', 'test', 'val',)
    })
    DATASETS.update({
        'gnn_LSUN_{}'.format(d): {
            'img_dir': 'LSUN_wf/images',
            'gnn_root': 'LSUN_wf/gnn_npz',
            'ann_file': 'LSUN_wf/{}.json'.format(d)}
        for d in ('train', 'test', 'val',)
    })
    DATASETS.update({
        'structured3D_opendoors_{}'.format(d): {
            'img_dir': 'Structured3D_wf_open_doors_1mm/images',
            'ann_file': 'Structured3D_wf_open_doors_1mm/{}.json'.format(d)}
        for d in ('train', 'train_mini', 'test', 'test_mini', 'val', 'val_mini')
    })
    DATASETS.update({
        'gnn_structured3D_opendoors_{}'.format(d): {
            'img_dir': 'Structured3D_wf_open_doors_1mm/images',
            'gnn_root': 'Structured3D_wf_open_doors_1mm/gnn_npz',
            'ann_file': 'Structured3D_wf_open_doors_1mm/{}.json'.format(d)}
        for d in ('train', 'train_mini', 'test', 'test_mini', 'val', 'val_mini')
    })
    DATASETS.update({
        'gnn_gt_structured3D_opendoors_{}'.format(d): {
            'img_dir': 'Structured3D_wf_open_doors_1mm/images',
            'gnn_root': 'Structured3D_wf_open_doors_1mm/gnn_npz_gt',
            'ann_file': 'Structured3D_wf_open_doors_1mm/{}.json'.format(d)}
        for d in ('train', 'train_mini', 'test', 'test_mini', 'val', 'val_mini')
    })
    DATASETS.update({
        'gnn_bmvc_structured3D_opendoors_{}'.format(d): {
            'img_dir': 'Structured3D_wf_open_doors_1mm/images',
            'gnn_root': 'Structured3D_wf_open_doors_1mm/gnn_bmvc_npz',
            'ann_file': 'Structured3D_wf_open_doors_1mm/{}.json'.format(d)}
        for d in ('train', 'train_mini', 'test', 'test_mini', 'val', 'val_mini')
    })

    @staticmethod
    def get(name):
        assert name in DatasetCatalog.DATASETS
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]

        args = dict(
            root = osp.join(data_dir,attrs['img_dir']),
            ann_file = osp.join(data_dir,attrs['ann_file'])
        )

        if 'gnn' in name:
            args['gnn_root'] = osp.join(data_dir,attrs['gnn_root'])
            return dict(factory="WireframeGNNDataset",args=args)

        if 'train' in name:
            return dict(factory="TrainDataset",args=args)
        if 'test' in name and 'ann_file' in attrs:
            return dict(factory="TestDatasetWithAnnotations",
                        args=args)
        if 'val' in name and 'ann_file' in attrs:
            return dict(factory="TestDatasetWithAnnotations",
                        args=args)
        raise NotImplementedError()
