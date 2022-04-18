import cv2
import torch
import torch.backends.cudnn as cudnn

import numpy as np
from argparse import Namespace
from pathlib import Path
from typing import List

# for load model
from detector.yolact_edge.utils.augmentations import (
    FastBaseTransform,
    BaseTransform,
)

from detector.yolact_edge.yolact import Yolact
from detector.yolact_edge.data import cfg, set_cfg
from detector.yolact_edge.utils.tensorrt import convert_to_tensorrt

from detector.yolact_edge.layers.output_utils import postprocess
from detector.tools.visualize_tools import create_image_with_objects


class YolactEdge:

    def __init__(
        self,
        params: dict,
    ) -> None:

        self.model_weights_path = params['weights']
        self.model_config = params['model_config']

        self.args = self.init_model_args(
            params['tensorrt'],
            params['score_threshold'],
        )
        set_cfg(self.args.config)

        # cuda
        cudnn.benchmark = True
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # load model
        self.net = Yolact(training=False)
        self.net.load_weights(self.model_weights_path, args=self.args)

        self.net.eval()
        
        convert_to_tensorrt(
            self.net,
            cfg,
            self.args,
            transform=BaseTransform(),
        )

        self.net.detect.use_fast_nms = self.args.fast_nms
        self.net = self.net.cuda()
        self.transform = FastBaseTransform()

        self.extras = {
            "backbone": "full",
            "interrupt": False,
            "keep_statistics": False,
            "moving_statistics": None,
        }

    def init_model_args(
        self,
        tensorrt: bool=True,
        score_threshold: float=0.2,
    ):
        return Namespace(
            ap_data_file='results/ap_data.pkl',
            bbox_det_file='results/bbox_detections.json',
            benchmark=True,
            calib_images=None,
            coco_transfer=False,
            config=self.model_config,
            crop=True,
            cuda=True,
            dataset=None,
            detect=False,
            deterministic=False,
            disable_tensorrt=not tensorrt,
            display=False,
            display_bboxes=True,
            display_lincomb=False,
            display_masks=True,
            display_scores=True,
            display_text=True,
            drop_weights=None,
            eval_stride=5,
            fast_eval=False,
            fast_nms=True,
            image='/home/user/Work/YolactEdge/yolact_edge/test_img.jpg:test_img_out.jpg',
            images=None,
            mask_det_file='results/mask_detections.json',
            mask_proto_debug=False,
            max_images=-1,
            no_bar=False,
            no_hash=False,
            no_sort=False,
            output_coco_json=False,
            output_web_json=False,
            resume=False,
            score_threshold=score_threshold,
            seed=None,
            shuffle=False,
            top_k=100,
            trained_model=self.model_weights_path,
            trt_batch_size=1,
            use_fp16_tensorrt=True,
            use_tensorrt_safe_mode=True,
            video=None,
            video_multiframe=1,
            web_det_path='web/dets/',
            yolact_transfer=False,
        )

    def get_classes_names(self):
        return cfg.dataset.class_names

    def inference(
        self,
        image: np.ndarray,
    ) -> List:

        '''
        Return int64, float32, int64, float32, 
        '''

        frame = torch.from_numpy(image).cuda().float()

        batch = self.transform(frame.unsqueeze(0))

        # prediction
        preds = self.net(batch, extras=self.extras)
        preds = preds["pred_outs"]

        h, w, _ = frame.shape

        labels, scores, boxes, masks = postprocess(
            det_output=preds,
            w=w,
            h=h,
            visualize_lincomb=self.args.display_lincomb,
            crop_masks=self.args.crop,
            score_threshold=self.args.score_threshold,
        )

        return labels, scores, boxes, masks, frame

    def inference_with_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        outputs = self.inference(image)

        output_image = create_image_with_objects(
            model_output=outputs,
            img=image,
            class_names=self.get_classes_names(),
        )

        return output_image


def read_opencv_image(image_path: str) -> np.ndarray:
    input_image = cv2.imread(image_path)
    return cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)


def save_opencv_image(image_path: str, image: np.ndarray):
    output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, output_image)


if __name__ == "__main__":

    # parameters
    model_weights_path = 'weights/yolact_edge_my_dataset_2_more_slow_trt_27_200000.pth'
    model_config = 'yolact_edge_my_dataset_2_more_slow_trt_config'

    # image
    input_image_path = 'data/test_img.jpg'
    out_image_path = 'data/test_img_out.jpg'

    # model
    model = YolactEdge(model_weights_path, model_config)

    # inference
    input_image = read_opencv_image(input_image_path)
    output_image = model.inference_with_image(input_image)
    save_opencv_image(out_image_path, output_image)
