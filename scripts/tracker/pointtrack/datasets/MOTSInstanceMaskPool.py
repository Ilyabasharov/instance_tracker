import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import cv2
import click
import torch
import numpy as np
from PIL import Image
import multiprocessing

from utils.file_utils import (
    make_dataset,
    remove_and_mkdir,
    save_pickle2,
)

ex=0.2
expand_ratio = [ex, ex]


SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]


def getImgLabel(img_path, label_path, label_id):
    img = cv2.imread(img_path)
    label = np.array(Image.open(label_path))
    mask = np.logical_and(label >= label_id * 1000, label < (label_id + 1) * 1000)
    label[(1-mask).astype(bool)] = 0
    return img, label


def get_grids(mask):
    # grid need to be in [-1,1], and uv rather than vu
    h, w = mask.shape
    s = torch.zeros(h)
    t = torch.zeros(h)
    for i in range(h):
        us = torch.nonzero(mask[i])
        if us.shape[0] > 0:
            t[i] = us.min()
            s[i] = (us.max() - us.min() + 1) / float(w)
        else:
            t[i] = 0
            s[i] = 1.0
    xTemplate = torch.arange(w).unsqueeze(0).repeat(h, 1)
    us = xTemplate * s.unsqueeze(-1) + t.unsqueeze(-1)
    vs = torch.arange(h).float().unsqueeze(-1).repeat(1, w)
    grid = torch.cat([us.unsqueeze(-1) / (w-1), vs.unsqueeze(-1) / (h-1)], dim=-1)
    return (grid - 0.5) * 2


def maskPool(info):
    name = info['name']
    outF = info['path_to_save']
    img, label = getImgLabel(info['img'], info['label'], info['label_id'])

    if np.unique(label).shape[0] == 1:
        return

    instList = []
    obj_ids = np.unique(label).tolist()[1:]
    for id in obj_ids:
        mask = (label == id).astype(np.uint8)
        maskX = (label > 0).astype(np.uint8) * 2
        maskX[mask > 0] = 1
        h, w = mask.shape
        vs, us = np.nonzero(mask)
        v0, v1 = vs.min(), vs.max()
        vlen = v1 - v0
        u0, u1 = us.min(), us.max()
        ulen = u1 - u0
        # enlarge box by expand_ratio
        v0 = max(0, v0 - int(expand_ratio[0] * vlen))
        v1 = min(v1 + int(expand_ratio[0]* vlen), h - 1)
        u0 = max(0, u0 - int(expand_ratio[1] * ulen))
        u1 = min(u1 + int(expand_ratio[1] * ulen), w - 1)
        inst_id = id
        sp = [v0, u0]
        imgCrop = img[v0:v1 + 1, u0:u1 + 1]
        maskCrop = mask[v0:v1 + 1, u0:u1 + 1]
        maskX = maskX[v0:v1 + 1, u0:u1 + 1]

        instList.append({'inst_id': inst_id, 'sp': sp, 'img': imgCrop, 'mask': maskCrop, 'maskX': maskX})

    save_pickle2(os.path.join(outF, name), instList)

def getPairs(
    id: str,
    path_to_dataset: str,
    path_to_save: str,
    label_id: int,
) -> None:

    label_root = os.path.join(path_to_dataset, 'instances', id)
    image_root = label_root.replace('instances', 'training/image_02')

    image_list = make_dataset(image_root, suffix='.png')
    image_list.sort()
    label_list = make_dataset(label_root, suffix='.png')
    label_list.sort()

    infos = []
    for ind, image_path in enumerate(image_list):
        infos.append({
            'name': id + '_' + str(ind) + '.pkl',
            'img': image_list[ind],
            'label': label_list[ind],
            'label_id': label_id,
            'path_to_save': path_to_save,
        })

    # decode all frames
    if len(infos) > 0:
        pool = multiprocessing.Pool(processes=32)
        pool.map(maskPool, infos)
        pool.close()

@click.command()
@click.option(
    '--input_path',
    default='../',
    help='Path to read the database.',
)
@click.option(
    '--output_path',
    default='../',
    help='Path to save the processed database.',
)
@click.option(
    '--car/--no-car',
    default=False,
    is_flag=True,
    help='Process data for cars.',
)
@click.option(
    '--pedestrian/--no-pedestrian',
    default=False,
    is_flag=True,
    help='Process data for pedestrians.',
)
def main(
    input_path: str,
    output_path: str,
    car: bool,
    pedestrian: bool,
) -> None:

    if car:
        outF = os.path.join(output_path, 'CarsEnvDB')
        remove_and_mkdir(outF)

        for i in SEQ_IDS_TRAIN:
            getPairs(i, input_path, outF, 1)
        for i in SEQ_IDS_VAL:
            getPairs(i, input_path, outF, 1)

    if pedestrian:
        outF = os.path.join(output_path, 'PedsEnvDB')
        remove_and_mkdir(outF)
        for i in SEQ_IDS_TRAIN:
            getPairs(i, input_path, outF, 2)
        for i in SEQ_IDS_VAL:
            getPairs(i, input_path, outF, 2)

if __name__ == '__main__':
    main()
