#!/usr/bin/python3

import os
import sys
import torch
import torch2trt
import collections

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from models.branchedERFNet import TrackerOffsetEmb

def get_embedding_model(
    params: dict,
    classname: str,
) -> TrackerOffsetEmb:

    # reconstruct the model from class
    model = TrackerOffsetEmb(
        device=params['device'],
        **params['offsetEmb'],
    ).to(params['device'])

    state = torch.load(
        f=params['weights'][classname],
        map_location='cpu',
    )

    if 'model_state_dict' in state:
        state = state['model_state_dict']

    # remove 'module.' of dataparallel
    new_state = collections.OrderedDict()
    for k in state:
        new_state[k[7:]] = state[k]

    model.load_state_dict(new_state)
    model.eval()

    if params['tensorrt']['convert']:

        print(f'Convert pointtrack to TRT for class {classname}...')

        weights_trt = params['weights'][classname] + '.trt'

        if os.path.exists(weights_trt):
            print('Loading TRT cache ...')
            model = torch2trt.TRTModule()
            model.load_state_dict(torch.load(weights_trt))

        else:

            model = torch2trt.torch2trt(
                module=model,
                inputs=[
                    # one batch required input
                    # points shape (B x NUM_POINTS x 8)
                    torch.zeros(
                        (1, 1500, 8),
                        dtype=torch.float32,
                        device=params['device'],
                    ),
                    # xyxys shape (B x 4)
                    torch.zeros(
                        (1, 4),
                        dtype=torch.float32,
                        device=params['device'],
                    ),
                ],
                max_batch_size=params['tensorrt']['max_batch_size'],
                fp16_mode=params['tensorrt']['fp16_mode'],
                int8_mode=params['tensorrt']['int8_mode'],
                input_names=['points', 'xyxys', ],
                output_names=['embedding', ],
            )

            torch.save(model.state_dict(), weights_trt)

        print('Done!')

    return model
