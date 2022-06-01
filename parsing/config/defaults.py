from yacs.config import CfgNode as CN
from .models import MODELS
from .dataset import DATASETS
from .solver import SOLVER
cfg = CN()

cfg.ENCODER = CN()
cfg.ENCODER.DIS_TH = 5
cfg.ENCODER.ANG_TH = 0.1
cfg.ENCODER.NUM_STATIC_POS_LINES = 300
cfg.ENCODER.NUM_STATIC_NEG_LINES = 40

cfg.MODEL = MODELS
cfg.DATASETS = DATASETS
cfg.SOLVER = SOLVER
cfg.CHECKPOINT = None
cfg.GNN_CHECKPOINT = None
cfg.TRANSFER_LEARN = False
cfg.SCORE_THRESHOLD = 0.6 # Threshold for the final line classification
cfg.LINE_NMS_THRESHOLD = -1
cfg.GRAPH_NMS = False

cfg.DATALOADER = CN()
cfg.DATALOADER.NUM_WORKERS = 8


# Used in WandB as well as for constructing path names for storing result and checkpoints.
cfg.EXPERIMENT = CN()
cfg.EXPERIMENT.NAME = None
cfg.EXPERIMENT.GROUP = None
cfg.EXPERIMENT.NOTES = None

cfg.OUTPUT_DIR = "outputs/dev"
