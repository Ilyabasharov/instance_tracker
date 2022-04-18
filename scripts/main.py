#!/usr/bin/python3
# coding: utf-8

import os
import cv2
import click
import yaml
import numpy as np
import tqdm
import time
import pycocotools.mask as mask_utils

from detector.yolact_edge_class import YolactEdge
from tracker.pointtrack_class import PointTrack

image_ext = ['jpg', 'jpeg', 'png', 'webp', ]
video_ext = ['mp4', 'mov', 'avi', 'mkv', ]


class Tracker:

    def __init__(
        self,
        config_path: str='../config.yaml',
    ) -> None:

        with open(config_path, 'r') as file:
            params = yaml.load(
                stream=file,
                Loader=yaml.FullLoader,
            )

        self.point_track = PointTrack(params['point_track'])
        self.yolact_edge = YolactEdge(params['yolact_edge'])

    def track(
        self,
        image: np.ndarray,
    ) -> tuple:

        labels, scores, boxes, masks, image = self.yolact_edge.inference(
            image=image,
        )

        labels, masks, track_ids = self.point_track.inference(
            image=image,
            masks=masks,
            boxes=boxes,
            labels=labels,
            scores=scores,
        )

        return labels, masks, track_ids

@click.command()
@click.option(
    '--input_video_path',
    default='/home/ilyabasharov/workspace/code/instance_tracker/data/kitti_mots/training/image_02/0020',
    help='Input video path.',
)
@click.option(
    '--config_path',
    default='../config.yaml',
    help='Config path for models.',
)
@click.option(
    '--save_results_as_txt',
    is_flag=True,
    default=False,
    help='Save results as KITTI MOTS format or not.',
)
@click.option(
    '--where_to_save',
    default='0020.txt',
    help='Where save results as txt.',
)
def main(
    input_video_path: str,
    config_path: str,
    save_results_as_txt: bool,
    where_to_save: str,
) -> None:
    
    for path in (config_path, input_video_path):
        assert os.path.exists(config_path), f'{path} does not exist!'

    if input_video_path[input_video_path.rfind('.') + 1:].lower() in video_ext:

        is_video = True
        capture = cv2.VideoCapture(input_video_path)
        length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    elif os.path.isdir(input_video_path):

        is_video = False
        image_names = []
        ls = os.listdir(input_video_path)

        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(input_video_path, file_name))
        
        length = len(image_names)

    else:

        raise NotImplementedError

    tracker = Tracker(config_path)

    counter = 0
    results = {}
    progress_bar = tqdm.tqdm(total=length)

    while True:
        
        ret = False

        if is_video:
            ret, frame = capture.read()
        else:
            if counter < len(image_names):
                frame = cv2.imread(image_names[counter])
                ret = True

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        tack = time.perf_counter()

        labels, masks, track_ids = tracker.track(frame)

        tick = time.perf_counter()

        if save_results_as_txt:

            instances = []

            for i in range(len(labels)):

                instance = {
                    'class_id': labels[i].item() + 1,
                    'track_id': track_ids[i].item(),
                    'segmentation': mask_utils.encode(np.asfortranarray(masks[i].cpu().numpy().astype(bool))),
                }

                instances.append(instance)

            results[counter] = instances

        counter += 1

        progress_bar.set_postfix({
            'num_objects': len(labels),
            'track_time': 1 / (tick - tack),
        })
        progress_bar.update()

    if save_results_as_txt:
        with open(where_to_save, 'w') as f:
            for tf in sorted(results.keys()):
                for i in range(len(results[tf])):
                    print(
                        tf,
                        results[tf][i]['class_id']*1000 + results[tf][i]['track_id'],
                        results[tf][i]['class_id'],
                        *results[tf][i]['segmentation']['size'],
                        results[tf][i]['segmentation']['counts'].decode(encoding='UTF-8'),
                        file=f,
                    )

if __name__ == '__main__':
    main()
