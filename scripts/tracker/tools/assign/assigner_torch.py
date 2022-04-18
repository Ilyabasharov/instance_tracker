from __future__ import annotations

import torch
import numpy as np
import collections

from torchvision.ops import box_iou
from utils import (
    mask_iou,
    center_mass,
    euclidean,
)

from scipy.optimize import linear_sum_assignment
#or
from tracker.tools.optim.hungarian import compute


TrackElement = collections.namedtuple(
    typename='TrackElement',
    field_names=['t', 'track_id', 'mask', 'box', 'embed', 'mean'],
)

class Assigner:

    def __init__(
        self,
        optim: str,
        alive_threshold: int,
        use_mask_iou: bool,
        use_bbox_iou: bool,
        iou_scale: float,
        iou_offset: float,
        euclidean_scale: float,
        euclidean_offset: float,
        association_threshold: float,
        means_threshold: float,
        *args,
        **kwargs,
    ) -> Assigner:

        self.use_mask_iou = use_mask_iou
        self.use_bbox_iou = use_bbox_iou

        self.iou_scale = iou_scale
        self.iou_offset = iou_offset

        self.euclidean_scale = euclidean_scale
        self.euclidean_offset = euclidean_offset

        self.association_threshold = association_threshold
        self.means_threshold = means_threshold
        self.alive_threshold = alive_threshold

        self.active_tracks = {
            'ts': [],
            'track_ids': [],
            'masks': [],
            'boxes': [],
            'embeds': [],
            'means': [],
        }
        self.next_inst_id = None

    def update_active_track(
        self,
        frame_count: int
    ) -> None:

        idx = np.where(self.active_tracks['ts'] >= frame_count - self.alive_threshold)[0]

        self.active_tracks['ts'] = self.active_tracks['ts'][idx]
        self.active_tracks['track_ids'] = self.active_tracks['track_ids'][idx]
        self.active_tracks['masks'] = self.active_tracks['masks'][idx]
        self.active_tracks['boxes'] = self.active_tracks['boxes'][idx]
        self.active_tracks['means'] = self.active_tracks['means'][idx]
        self.active_tracks['embeds'] = self.active_tracks['embeds'][idx]

    @torch.no_grad()
    def tracking(
        self,
        frame_count: int,
        embeds: torch.Tensor,
        masks: torch.Tensor,
        boxes: torch.Tensor,
    ) -> list:
    
        ''' Main step of Assigner
        
        Inputs:
        frame_count: int. Like abstract time,
        embeds: torch.Tensor. Shape: (Nx128), dtype: float
        masks: torch.Tensor. Shape: (NxHxW), dtype: float
        boxes: torch.Tensor. Shape: (Nx4), dtype: float
        
        Ouputs:
        result: torch.Tensor, dtype: int32
        '''
        
        assert len(embeds) == len(masks), 'Number of objects should be the same'

        if self.next_inst_id is None:
            self.next_inst_id = 1
        else:
            self.update_active_track(frame_count)
        
        n = len(embeds)
        
        if n == 0:
            result = torch.zeros(
                size=(n, ),
                dtype=torch.int32,
                device=embeds.device,
            )

            return result
        
        means = center_mass(masks)
        
        if len(self.active_tracks['ts']) == 0:

            self.active_tracks['ts'] = np.full((n, ), frame_count)
            self.active_tracks['track_ids'] = np.arange(self.next_inst_id, self.next_inst_id + n)
            self.active_tracks['masks'] = masks
            self.active_tracks['boxes'] = boxes
            self.active_tracks['means'] = means
            self.active_tracks['embeds'] = embeds
            
            result = torch.arange(
                start=self.next_inst_id,
                end=self.next_inst_id + n,
                device=masks.device,
                dtype=torch.int32,
            )

            self.next_inst_id += n

            return result

        result = torch.zeros(
            size=(n, ),
            dtype=torch.int32,
            device=embeds.device,
        )
        
        # compare inst by inst.
        
        # array for assigned instances
        detections_unassigned = torch.ones(
            size=(n, ),
            dtype=torch.bool,
            device=masks.device,
        )
        
        # step 1. Use distance between embeddings
        asso_sim = self.euclidean_scale * (
            self.euclidean_offset - torch.cdist(
                x1=embeds.unsqueeze(0),
                x2=self.active_tracks['embeds'].unsqueeze(0),
                p=2.,
            ).squeeze(0)
        )
        
        # step 2. Use distance between masks (IoU)
        if self.use_mask_iou:
            asso_sim += self.iou_scale * (
                self.iou_offset + mask_iou(
                    mask1=masks,
                    mask2=self.active_tracks['masks'],
                )
            )

        # step 3. Use distance between boxes (IoU)
        if self.use_bbox_iou:
            asso_sim += self.iou_scale * (
                self.iou_offset + box_iou(
                    boxes1=boxes,
                    boxes2=self.active_tracks['boxes'],
                )
            )

        # step 4. make some connections unsolvable
        idx, idy = (asso_sim <= self.association_threshold).nonzero(as_tuple=True)

        # prevent torch2trt error
        cost_mat = asso_sim.max() - asso_sim
        cost_mat[idx.cpu().numpy(), idy.cpu().numpy()] = 1e9
        
        # step 5. solve matrix
        rows, cols = linear_sum_assignment(
            cost_matrix=cost_mat.cpu().numpy(),
            maximize=False,
        )

        # to add this in active tracks
        ts_new = []
        track_ids_new = []
        masks_new = []
        boxes_new = []
        means_new = []
        embeds_new = []

        for row, column in zip(rows, cols):
        
            # the instance was the same
            if euclidean(self.active_tracks['means'][column], means[row]) < self.means_threshold \
             and cost_mat[row][column] != 1e9:

                self.active_tracks['ts'][column] = frame_count
                self.active_tracks['masks'][column] = masks[row]
                self.active_tracks['boxes'][column] = boxes[row]
                self.active_tracks['means'][column] = means[row]
                self.active_tracks['embeds'][column] = embeds[row]

                result[row] = self.active_tracks['track_ids'][column]
            
            # it is a new instance
            else:

                ts_new.append(frame_count)
                track_ids_new.append(self.next_inst_id)
                masks_new.append(masks[row])
                boxes_new.append(boxes[row])
                means_new.append(means[row])
                embeds_new.append(embeds[row])

                result[row] = self.next_inst_id
                self.next_inst_id += 1

            detections_unassigned[row] = False

        # new track id for unassigned instances
        
        for i in detections_unassigned.nonzero(as_tuple=True)[0]:

            ts_new.append(frame_count)
            track_ids_new.append(self.next_inst_id)
            masks_new.append(masks[i])
            boxes_new.append(boxes[i])
            means_new.append(means[i])
            embeds_new.append(embeds[i])
            
            result[i] = self.next_inst_id
            self.next_inst_id += 1

        if len(ts_new) == 0:
            return result

        self.active_tracks['ts'] = np.hstack(
            [
                self.active_tracks['ts'],
                np.stack(ts_new),
            ]
        )

        self.active_tracks['track_ids'] = np.hstack(
            [
                self.active_tracks['track_ids'],
                np.stack(track_ids_new),
            ]
        )

        self.active_tracks['boxes'] = torch.cat(
            [
                self.active_tracks['boxes'],
                torch.stack(boxes_new),
            ],
            dim=0,
        )

        self.active_tracks['masks'] = torch.cat(
            [
                self.active_tracks['masks'],
                torch.stack(masks_new),
            ],
            dim=0,
        )

        self.active_tracks['embeds'] = torch.cat(
            [
                self.active_tracks['embeds'],
                torch.stack(embeds_new),
            ],
            dim=0,
        )

        self.active_tracks['means'] = torch.cat(
            [
                self.active_tracks['means'],
                torch.stack(means_new),
            ],
            dim=0,
        )

        return result