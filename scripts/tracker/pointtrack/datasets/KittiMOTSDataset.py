import glob
import os
import sys
import random
import numpy as np
import math
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from utils.file_utils import (
    make_dataset,
    load_pickle,
)


class MOTSTrackCarsTrain(Dataset):

    SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
    SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
    TIMESTEPS_PER_SEQ = {
        "0000": 154, "0001": 447, "0002": 233, "0003": 144, "0004": 314, "0005": 297, "0006": 270,
        "0007": 800, "0008": 390, "0009": 803, "0010": 294, "0011": 373, "0012": 78, "0013": 340,
        "0014": 106, "0015": 376, "0016": 209, "0017": 145, "0018": 339, "0019": 1059, "0020": 837,
    }

    def __init__(
        self,
        database_path: str,
        dataset_root: str,
        type: str='train',
        num_points: int=250,
        transform=None,
        shift: bool=False,
        sample_num: int=30,
        nearby: int=1,
        trainval: bool=False,
        category_embedding: list=[
            [0.9479751586914062, 0.4561353325843811, 0.16707628965377808],
            [0.1,                -0.1,               0.1                ],
            [0.5455077290534973, -0.6193588972091675, -2.629554510116577,],
            [-0.1,                0.1,               -0.1,              ],
        ],
    ) -> None:

        self.type = 'training' if type in 'training' else 'testing'
        if trainval:
            self.squence = self.SEQ_IDS_TRAIN + self.SEQ_IDS_VAL
            print('Train with training and val set')
        else:
            self.squence = self.SEQ_IDS_TRAIN if self.type == 'training' else self.SEQ_IDS_VAL

        self.transform = transform

        self.dbDict = {}
        for id in self.squence:
            image_root = os.path.join(dataset_root, 'training/image_02', id)
            image_list = make_dataset(image_root, suffix='.png')
            image_list.sort()
            infos = {}
            for ind, image_path in enumerate(image_list):
                pkl_path = os.path.join(database_path, id + '_' + str(ind) + '.pkl')
                if os.path.isfile(pkl_path):
                    infos[ind] = load_pickle(pkl_path)
            self.dbDict[id] = infos

        self.mots_car_instances = self.getInstanceFromDB(self.dbDict)
        print('dbDict Loaded, %s instances' % len(self.mots_car_instances))

        self.inst_names = list(self.mots_car_instances.keys())
        self.inst_num = len(self.inst_names)
        self.mots_class_id = 1
        self.vMax, self.uMax = 375.0, 1242.0
        self.offsetMax = 128.0
        self.num_points = num_points
        self.shift = shift
        self.frequency = 1
        self.sample_num = sample_num
        self.nearby = nearby
        self.category_embedding = np.array(category_embedding, dtype=np.float32)

        print('MOTS Dataset created')

    def getInstanceFromDB(self, dbDict):
        allInstances = {}
        for k, fs in dbDict.items():
            # current video k
            # num_frames = self.TIMESTEPS_PER_SEQ[k]
            if not k in self.squence:
                continue
            for fi, f in fs.items():
                frameCount = fi
                for inst in f:
                    inst_id = k + '_' + str(inst['inst_id'])
                    newDict = {'frame': frameCount, 'sp': inst['sp'], 'img': inst['img'], 'mask': inst['mask'], 'maskX': inst['maskX']}
                    if not inst_id in allInstances.keys():
                        allInstances[inst_id] = [newDict]
                    else:
                        allInstances[inst_id].append(newDict)
        return allInstances

    def __len__(self):
        return len(self.inst_names)

    def get_data_from_mots(self, index):
        # sample ? instances from self.inst_names
        inst_names_inds = random.sample(range(len(self.inst_names)), self.sample_num)
        inst_names = [self.inst_names[el] for el in inst_names_inds]
        pickles = [self.mots_car_instances[el] for el in inst_names]

        sample = {}
        sample['mot_im_name0'] = index
        sample['points'] = []
        sample['labels'] = []
        sample['imgs'] = []
        sample['inds'] = []
        sample['envs'] = []
        sample['xyxys'] = []
        for pind, pi in enumerate(pickles):
            inst_id = pind + 1
            inst_length = len(pi)
            if inst_length > 2:
                mid = random.choice(range(1, inst_length - 1))
                nearby = random.choice(range(1, self.nearby+1))
                start, end = max(0, mid - nearby), min(inst_length - 1, mid + nearby)
                pis = [pi[start], pi[mid], pi[end]]
            else:
                start, end = 0, 1
                pis = pi[start:end + 1]
            for ii, inst in enumerate(pis):
                img = inst['img']
                mask = inst['mask'].astype(bool)
                maskX = inst['maskX']
                sp = inst['sp']
                assert (~mask).sum() > 0

                ratio = 2.0
                bg_num = int(self.num_points / (ratio + 1))
                fg_num = self.num_points - bg_num

                # get center
                vs_, us_ = np.nonzero(mask)
                vc, uc = vs_.mean(), us_.mean()

                vs, us = np.nonzero(~mask)
                vs = (vs - vc) / self.offsetMax
                us = (us - uc) / self.offsetMax
                rgbs = img[~mask] / 255.0
                if self.shift:
                    # us += (random.random() - 0.5) * 0.05  # -0.025~0.025
                    vs += np.random.normal(0, 0.001, size=vs.shape)  # random jitter
                    us += np.random.normal(0, 0.001, size=us.shape)  # random jitter
                cats = maskX[~mask]
                cat_embds = self.category_embedding[cats]
                pointUVs = np.concatenate([rgbs, vs[:, np.newaxis], us[:, np.newaxis], cat_embds], axis=1)
                choices = np.random.choice(pointUVs.shape[0], bg_num)
                points_bg = pointUVs[choices][np.newaxis, :, :].astype(np.float32)

                vs = (vs_ + sp[0]) / self.vMax
                us = (us_ + sp[1]) / self.uMax  # to compute the bbox position
                sample['xyxys'].append([us.min(), vs.min(), us.max(), vs.max()])

                vs = (vs_ - vc) / self.offsetMax
                us = (us_ - uc) / self.offsetMax
                rgbs = img[mask.astype(bool)] / 255.0
                if self.shift:
                    # us += (random.random() - 0.5) * 0.05  # -0.025~0.025
                    vs += np.random.normal(0, 0.001, size=vs.shape)  # random jitter
                    us += np.random.normal(0, 0.001, size=us.shape)  # random jitter
                pointUVs = np.concatenate([rgbs, vs[:, np.newaxis], us[:, np.newaxis]], axis=1)
                choices = np.random.choice(pointUVs.shape[0], fg_num)
                pointUVs = np.concatenate([rgbs, vs[:, np.newaxis], us[:, np.newaxis]], axis=1)
                points_fg = pointUVs[choices][np.newaxis, :, :].astype(np.float32)
                points_fg = np.concatenate(
                    [points_fg, np.zeros((points_fg.shape[0], points_fg.shape[1], 3), dtype=np.float32)], axis=-1)

                sample['points'].append(np.concatenate([points_fg, points_bg], axis=1))
                sample['labels'].append(np.array(inst_id)[np.newaxis])
                sample['envs'].append(fg_num)

        sample['points'] = np.concatenate(sample['points'], axis=0)
        sample['envs'] = np.array(sample['envs'], dtype=np.int32)
        sample['labels'] = np.concatenate(sample['labels'], axis=0)
        sample['xyxys'] = np.array(sample['xyxys'], dtype=np.float32)
        return sample

    def __getitem__(self, index):
        # select nearby images from mots
        while 1:
            try:
                sample = self.get_data_from_mots(index)
                break
            except:
                pass
        # sample = self.get_data_from_mots(index)

        # transform
        if (self.transform is not None):
            sample = self.transform(sample)
            return sample
        else:
            return sample
