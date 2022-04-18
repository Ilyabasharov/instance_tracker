import cv2
import torch
import argparse
import numpy as np
from pathlib import Path

from yolact_edge_class import YolactEdge, read_opencv_image
from tools.benchmark_tools import Timer


timer = Timer(verbose=True)

@timer.timeit
def run_model_on_frame(model: YolactEdge, frame: np.ndarray):
    frame = torch.from_numpy(frame).cuda().float()
    model.inference(frame) 


def execute_images(images_path: str, model: YolactEdge):
    images_path = Path(images_path)
    all_imgs_paths = list(images_path.glob('*'))
    for img_path in all_imgs_paths:
        image = read_opencv_image(str(img_path))
        run_model_on_frame(model, image)
        

def execute_video(video_path: str, model: YolactEdge):   
    capture = cv2.VideoCapture(video_path)
    if capture.isOpened():
        print('Video is opened')
        while capture.isOpened():
            ret, frame = capture.read()      
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                run_model_on_frame(model, frame)
            else:
                break


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weights_path', default='weights/yolact_edge_my_dataset_2_more_slow_trt_27_200000.pth', type=str, help='Path to model weights \(.pth\)')
    parser.add_argument('--model_config', default='yolact_edge_my_dataset_2_more_slow_trt_config', type=str, help='Model config name')
    parser.add_argument('--frame_source_type', choices=['video', 'images'], default='video', type=str, help='Further restrict the number of predictions to parse')
    parser.add_argument('--frame_source_path', default='./data/invalid.mp4', type=str, help='Path to image folder or video file to execute model on')
    return parser


if __name__ == "__main__":
    
    parser = init_parser()
    args = parser.parse_args()
       
    # model
    model = YolactEdge(args.model_weights_path, args.model_config)
    
    if args.frame_source_type == 'images':
        execute_images(args.frame_source_path, model)
    else:
        execute_video(args.frame_source_path, model)
            
    print('Execution time: ')
    timer.display()        


