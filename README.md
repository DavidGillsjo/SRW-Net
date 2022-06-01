# Semantic Room Wireframe Detection
This repo contains code to generate Semantic Room Wireframe (SRW) annotations from
Structured3D and LSUN. It also holds the implementation of SRW-Net.

## Clone repo and get submodules
```
git clone --recurse-submodules git@github.com:DavidGillsjo/semantic-room-wireframe.git
```
alternatively
```
git clone git@github.com:DavidGillsjo/semantic-room-wireframe.git
git submodule init
git submodule update
```

## Docker
We supply a [Dockerfile](docker/Dockerfile) to build a docker image which can run the code.
First, modify [line 7](docker/Dockerfile#L7) so that `gpu_arch` matches your GPU architecture. See for example [this blog post](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) to find your arch code.

Then build and run:
```
cd docker
./build_docker.sh
./run_docker.sh
```
You will find your `HOME` directory mounted to `/host_home`.

## Fix Python path
Add `parsing` folder to python path for correct imports.
```
source init_env.sh
```

## Download Annotations and Model Weights
Here you find the annotated data together with the pre-trained models to
reproduce the result from the paper. We also include the pretrained [HAWP](https://github.com/cherubicXN/hawp) model which we used as initialization for the training.
- [Original HAWP model](https://vision.maths.lth.se/davidg-data/srw-net/models/model_hawp.pth)
- [SRW-Net Predictor CNN](https://vision.maths.lth.se/davidg-data/srw-net/models/model_proposal_s3d.pth)
- [SRW-Net Refinement GCN](https://vision.maths.lth.se/davidg-data/srw-net/models/model_gnn_s3d.pth)
- [Annotations](https://vision.maths.lth.se/davidg-data/srw-net/data.zip)

Unzip `data.zip` to the `data` folder.
You may also put the model weights in the `data` folder, the rest of this README will assume you did.

## Generate Annotations
If you prefer to generate the annotations, then
- Download Structured3D, see [official website](https://structured3d-dataset.org/).
You may use [this script](data/download_structured3D.py).
- Run `python3 preprocessing/structured3D2wireframe.py --help` for instructions.


## Build
To run the network, some C-code needs compiling.
```
./build.sh
```

## Inference
There are a number of ways to run inference, see `python3 scripts/test.py --help` for details.

### Run inference on test set
To run on the test set, do
```
cd scripts
python3 test.py \
--config-file ../config/layout-SRW-S3D.yaml \
CHECKPOINT ../data/model_proposal_s3d.pth
GNN_CHECKPOINT ../data/model_gnn_s3d.pth
OUTPUT_DIR ../runs/test
```
To run on the validation data, add the flag `--val`.

### Run inference on your own images
To run on a set of images
```
cd scripts
python3 test.py \
--config-file ../config/layout-SRW-S3D.yaml \
--img-folder <my-image-folder> \
CHECKPOINT ../data/model_proposal_s3d.pth
GNN_CHECKPOINT ../data/model_gnn_s3d.pth
```
and the result will be placed in `<my-image-folder>/test`.

## Train
The Predictor and the Refinement module are trained separately in the following steps

### Train Predictor from the HAWP model weights
```
cd scripts
python3 train.py \
--config-file ../config/Pred-SRW-S3D.yaml \
CHECKPOINT ../data/model_hawp.pth \
OUTPUT_DIR ../runs/predictor \
TRANSFER_LEARN True
```
Model checkpoints will be in `runs/predictor/<datetime>/model_<epoch>.pth`.

To monitor the training you may start a tensorboard instance (also a docker container)
```
./start_tensorboard.sh
```
Then open your browser and go to http://localhost:6006.

### Generate intermediate dataset
Now we use the trained model to generate a dataset for the Refinement module.
```
cd preprocessing
python3 generate_gnn_dataset.py \
--config-file ../config/Pred-SRW-S3D.yaml \
--output-dir ../data/Structured3D_wf_open_doors_1mm/gnn_npz \
CHECKPOINT ../runs/predictor/<datetime>/model_39.pth \
IMS_PER_BATCH 1
```

### Train the Refinement Module
Finally, we train the Refinement GCN.
```
cd scripts
python3 train.py \
--config-file ../config/GNN-SRW-S3D.yaml \
OUTPUT_DIR ../runs/refinement
```
and you will find the model weights at `../runs/refinement/<datetime>/model_09.pth`.

## Citation
If you use it in your research, please cite
```
```
