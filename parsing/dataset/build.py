import torch
from .transforms import *
from . import train_dataset
from parsing.config.paths_catalog import DatasetCatalog
from . import test_dataset
from . import gnn_dataset

def build_transform(cfg):
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                                           cfg.DATASETS.IMAGE.PIXEL_STD,
                                           cfg.DATASETS.IMAGE.TO_255)
        ]
    )
    return transforms

def build_train_dataset(cfg):
    assert len(cfg.DATASETS.TRAIN) == 1
    name = cfg.DATASETS.TRAIN[0]
    dargs = DatasetCatalog.get(name)

    factory = getattr(train_dataset,dargs['factory'])
    args = dargs['args']
    args['transform'] = Compose(
                                [Resize(cfg.DATASETS.IMAGE.HEIGHT,
                                        cfg.DATASETS.IMAGE.WIDTH,
                                        cfg.DATASETS.TARGET.HEIGHT,
                                        cfg.DATASETS.TARGET.WIDTH),
                                 ReMapLabels(cfg.DATASETS.LINE_CLASS_TYPE,
                                             cfg.DATASETS.DISABLE_CLASSES),
                                 ToTensor(),
                                 Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                                           cfg.DATASETS.IMAGE.PIXEL_STD,
                                           cfg.DATASETS.IMAGE.TO_255)])
    args['hflip'] = cfg.DATASETS.HFLIP
    args['vflip'] = cfg.DATASETS.VFLIP
    dataset = factory(**args)

    dataset = torch.utils.data.DataLoader(dataset,
                                          batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                          collate_fn=train_dataset.collate_fn,
                                          shuffle = True,
                                          num_workers = cfg.DATALOADER.NUM_WORKERS)
    return dataset

def build_test_dataset(cfg, validation=False):
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         ReMapLabels(cfg.DATASETS.LINE_CLASS_TYPE,
                     cfg.DATASETS.DISABLE_CLASSES),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                   cfg.DATASETS.IMAGE.PIXEL_STD,
                   cfg.DATASETS.IMAGE.TO_255)
        ]
    )

    datasets = []
    dset_list = cfg.DATASETS.VAL if validation else cfg.DATASETS.TEST
    for name in dset_list:
        dargs = DatasetCatalog.get(name)
        factory = getattr(test_dataset,dargs['factory'])
        args = dargs['args']
        args['transform'] = transforms
        dataset = factory(**args)
        dataset = torch.utils.data.DataLoader(
            dataset,  batch_size = 1,
            collate_fn = dataset.collate_fn,
            num_workers = cfg.DATALOADER.NUM_WORKERS,
        )
        datasets.append((name,dataset))
    return datasets

def build_generate_dataset(cfg):
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                   cfg.DATASETS.IMAGE.PIXEL_STD,
                   cfg.DATASETS.IMAGE.TO_255)
        ]
    )

    datasets = []
    dset_list = cfg.DATASETS.TRAIN + cfg.DATASETS.VAL + cfg.DATASETS.TEST
    for name in dset_list:
        dargs = DatasetCatalog.get(name)
        factory = test_dataset.TestDatasetWithAnnotations
        args = dargs['args']
        args['transform'] = transforms
        dataset = factory(**args)
        dataset = torch.utils.data.DataLoader(
            dataset,  batch_size = 1,
            collate_fn = dataset.collate_fn,
            num_workers = cfg.DATALOADER.NUM_WORKERS,
        )
        datasets.append((name,dataset))
    return datasets

def build_gnn_train_dataset(cfg):

    assert len(cfg.DATASETS.TRAIN) == 1
    name = cfg.DATASETS.TRAIN[0]
    dargs = DatasetCatalog.get(name)

    factory = getattr(gnn_dataset,dargs['factory'])
    args = dargs['args']
    args['transform'] = Compose([
         ToTensorGNN(),
    ])
    args['ann_transform'] =  Compose([
         ReMapLabels(cfg.DATASETS.LINE_CLASS_TYPE,
                     cfg.DATASETS.DISABLE_CLASSES),
    ])

    args['training'] = True
    dataset = factory(**args)

    dataset = gnn_dataset.WireframeGNNDataloader(dataset,
                                          batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                          # shuffle = True,
                                          shuffle = False,
                                          num_workers = cfg.DATALOADER.NUM_WORKERS)
    return dataset

def build_gnn_test_dataset(cfg, validation=False):

    datasets = []
    dset_list = cfg.DATASETS.VAL if validation else cfg.DATASETS.TEST
    for name in dset_list:
        dargs = DatasetCatalog.get(name)
        factory = getattr(gnn_dataset,dargs['factory'])
        args = dargs['args']
        args['transform'] = ToTensorGNN()
        args['training'] = False
        args['ann_transform'] =  Compose([
             ReMapLabels(cfg.DATASETS.LINE_CLASS_TYPE,
                         cfg.DATASETS.DISABLE_CLASSES),
             # ResizeAnn(cfg.DATASETS.TARGET.HEIGHT,
             #           cfg.DATASETS.TARGET.WIDTH),
        ])
        dataset = factory(**args)
        dataset = gnn_dataset.WireframeGNNDataloader(
            dataset,  batch_size = 1,
            num_workers = 0,
        )
        datasets.append((name,dataset))
    return datasets
