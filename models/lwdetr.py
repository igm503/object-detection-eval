import torch
import onnxruntime as ort
import numpy as np

from ..eval import Detection

MEAN = (255 * np.array([0.485, 0.456, 0.406]))[:, None, None]
STD = (255 * np.array([0.229, 0.224, 0.225]))[:, None, None]


class LWDETR:
    def __init__(self, path: str, device: str):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(path, sess_options, providers=providers)

    def __call__(self, input):
        """
        Only supports single frame input
        """
        input = input.cpu().numpy()
        input = (input - MEAN) / STD
        input = input.astype(np.float32)
        output = self.session.run(None, {"input": input})

        logits, out_bbox = torch.Tensor(output[1]), torch.Tensor(output[0])

        prob = logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(logits.shape[0], -1), 300, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // logits.shape[2]
        labels = topk_indexes % logits.shape[2]
        boxes = self.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        boxes[:, :, (0, 2)] *= input.shape[1]
        boxes[:, :, (1, 3)] *= input.shape[0]

        detections = []
        for bbox, class_id, score in zip(
            boxes[0].cpu().numpy(), labels[0].cpu().numpy(), scores[0].cpu().numpy()
        ):
            detections.append(Detection(class_id, bbox, score))

        return detections

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [
            (x_c - 0.5 * w.clamp(min=0.0)),
            (y_c - 0.5 * h.clamp(min=0.0)),
            (x_c + 0.5 * w.clamp(min=0.0)),
            (y_c + 0.5 * h.clamp(min=0.0)),
        ]
        return torch.stack(b, dim=-1)
