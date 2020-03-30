# -*- coding: utf-8 -*-

import argparse
import logging
import random
import cv2
import torch

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

from config import add_config
from modeling import build_retinanet_efficientnet_bifpn_backbone

logging.basicConfig(level=logging.INFO)


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        assert "instances" in predictions
        instances = predictions["instances"].to(self.cpu_device)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            assert "instances" in predictions
            predictions = predictions["instances"].to(self.cpu_device)
            vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        for frame in frame_gen:
            yield process_predictions(frame, self.predictor(frame))


def setup_cfg(config_file_path: str, model_file_path: str, score_thresh: float = 0.5):
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(config_file_path)
    cfg.MODEL.WEIGHTS = model_file_path
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    cfg.freeze()
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--model-file", required=True, type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config_file_path = args.config_file
    model_file_path = args.model_file
    cfg = setup_cfg(config_file_path, model_file_path, score_thresh=0.5)
    demo = VisualizationDemo(cfg)

    dataset = DatasetCatalog.get(cfg.DATASETS.TEST[0])
    for _ in range(10):
        image_path = random.choice(dataset)["file_name"]
        image = cv2.imread(image_path)
        predictions, vis_output = demo.run_on_image(image)
        cv2.imshow("show", vis_output.get_image()[:, :, ::-1])
        cv2.waitKey()


if __name__ == "__main__":
    main()
