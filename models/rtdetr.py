import onnxruntime as ort
import numpy as np

from ..eval import Detection


class RTDETR:
    def __init__(self, path: str, device: str):
        ort.set_default_logger_severity(3)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(path, sess_options, providers=providers)

    def __call__(self, images):
        images = images.cpu().numpy()
        orig_sizes = np.array([np.array([640, 640]) for _ in range(len(images))])
        images = images.astype(np.float32).transpose(0, 1, 2, 3)
        outputs = self.session.run(None, {"images": images, "orig_target_sizes": orig_sizes})
        batch_classes = outputs[0]
        batch_bboxes = outputs[1]
        batch_confs = outputs[2]
        detections = []
        for class_ids, bboxes, confs in zip(batch_classes, batch_bboxes, batch_confs):
            dets = [
                Detection(int(class_id), bbox, conf)
                for class_id, bbox, conf in zip(class_ids, bboxes, confs)
            ]
            detections.append(dets)
        return detections
