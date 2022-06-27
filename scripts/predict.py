import json
import sys
sys.path.append("..")
from cog import BaseModel, BasePredictor, Input, Path
import torch
from parsing.config import cfg
from parsing.utils.comm import to_device

import os.path as osp
from skimage import io
import matplotlib.pyplot as plt
import json
import numpy as np
from scripts.test import ModelTester


class Output(BaseModel):
    plot: Path = None
    Json: str = None


class Predictor(BasePredictor):
    def setup(self):
        cfg.merge_from_file('config-files/layout-SRW-S3D.yaml')
        cfg.freeze()
        self.tester = ModelTester(cfg)
        self.model, checkpointer = self.tester._init_model()
        checkpointer.load(f='data/model_proposal_s3d.pth', f_gnn='data/model_gnn_s3d.pth', use_latest=False)
        self.model.eval()

    def predict(
        self,
        input_image: Path = Input(description="Image to be classified"),
        output_format: str = Input(description="Recieve outputs as Json or a debug-plot",
            default="Plot",
            choices=["Json", "Plot"],
        ),
        confidence_threshold: float = Input(default=0.9),
        # ignore_invalid_junc: bool = Input(default=True, description="Force junctions to have proper labels"),
    ) -> Output:
        with torch.no_grad():
            image_int = io.imread(str(input_image))
            image_tensor = self.tester.img_transform(image_int.astype(float)[:, :, :3])
            ann = {
                'height': image_int.shape[0],
                'width': image_int.shape[1],
                'filename':  osp.basename(str(input_image))
            }
            with torch.no_grad():
                output, extra_info = self.model(image_tensor[None].cuda(), [ann])
                output = to_device(output, torch.device('cpu'))
                self.tester.run_nms(output)

            if output_format == "Json":
                result_dict = create_json_output(output, 0.9, self.tester.cfg.MODEL.LINE_LABELS, self.tester.cfg.MODEL.JUNCTION_LABELS)
                return Output(Json=json.dumps(result_dict, indent=4))
            else:
                output_path = "output.png"
                self.tester.img_viz.plot_final_pred(image_int, 'output', output, float(confidence_threshold), ignore_invalid_junc=True, show_legend=True)
                plt.savefig(output_path)
                return Output(plot=Path(output_path))


def create_json_output(model_output, score_threshold, line_label_names, junction_label_names):
    result = {"Lines": [], "Junctions": []}
    lines_maks = ((model_output['lines_label_score'] > score_threshold) & (model_output['lines_label'] != 0)).numpy()
    lines = model_output['lines_pred'][lines_maks].numpy()
    lines_label = model_output['lines_label'][lines_maks].numpy()
    for i in range(len(lines)):
        line = {"type": line_label_names[lines_label[i]], "x1": float(lines[i][0]), "y1": float(lines[i][1]), "x2": float(lines[i][2]), "y2": float(lines[i][3])}
        result["Lines"].append(line)

    l2j_idx = model_output['line2junc_idx'][lines_maks].numpy()
    junctions = model_output['juncs_pred'].numpy()
    junctions_all_scores = model_output['juncs_score'].numpy()
    junctions_label = model_output['juncs_label'].numpy()
    invalid_mask = junctions_label == 0
    junctions_label[invalid_mask] = np.argmax(junctions_all_scores[invalid_mask, 1:], axis=1) + 1

    for j in np.unique(l2j_idx):
        junction = {"type": junction_label_names[junctions_label[j]], "x": float(junctions[j][0]), "y": float(junctions[j][1])}
        result["Junctions"].append(junction)
    return result


