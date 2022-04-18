#!/usr/bin/python3
from __future__ import annotations

import torch
import numpy as np
import numba as nb
import collections
import pycocotools.mask as maskUtils

from scipy.spatial.distance import (
    cdist,
    euclidean,
)

from scipy.optimize import linear_sum_assignment


'''
t - frame count
id - position in input data
track_id - assigned id
class_id - class id
mask - instance mask in np.array style
embed - embedding in np.array style
mean - center of the object in np.array style
'''

TrackElement = collections.namedtuple(
    typename='TrackElement',
    field_names=['t', 'id', 'track_id', 'mask', 'embed', 'mean'],
)

@nb.jit(
    nopython=True,
    fastmath=True,
    parallel=True,
    nogil=True,
    cache=True,
)
def center_mass(
    masks: np.array,
) -> np.array:
    
    '''
    Compute center mass in pixels for N masks
    Inputs:
    masks: np.array. Shape: NxHxW.
    Outputs:
    center_mass: np.array. Shape: Nx2.
    '''
    
    N = masks.shape[0]
    
    center_mass = np.empty(
        shape=(N, 2),
        dtype=np.float32,
    )

    for i in nb.prange(N):
        
        xs, ys = masks[i].nonzero()
        
        center_mass[i, 0] = xs.mean()
        center_mass[i, 1] = ys.mean()
        
    return center_mass

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
        
        self.optim = optim
        self.use_mask_iou = use_mask_iou
        self.use_bbox_iou = use_bbox_iou

        self.iou_scale = iou_scale
        self.iou_offset = iou_offset

        self.euclidean_scale = euclidean_scale
        self.euclidean_offset = euclidean_offset

        self.association_threshold = association_threshold
        self.means_threshold = means_threshold
        self.alive_threshold = alive_threshold

        self.active_tracks = []
        self.next_inst_id = None

    def update_active_track(
        self,
        frame_count: int
    ) -> None:

        self.active_tracks = [
            track for track in self.active_tracks
            if track.t >= frame_count - self.alive_threshold
        ]

    def tracking(
        self,
        frame_count: int,
        embeds: np.array,
        masks: np.array,
        **kwargs,
    ) -> list:
    
        ''' Main step of Assigner
        
        Inputs:
        frame_count: int. Like abstract time,
        embeds: np.array. Shape: (Nx128), dtype: float
        masks: np.array. Shape: (NxHxW), dtype: bool
        
        Ouputs:
        result: list. List with assigned indexes for each instance in the same order as input data
        '''
        
        assert len(embeds) == len(masks)

        if self.next_inst_id is None:
            self.next_inst_id = 1
        else:
            self.update_active_track(frame_count)
        
        n = len(embeds)

        result = np.zeros(
            shape=(n, ),
            dtype=np.int32,
        )
        
        if n < 1:
            return result
        
        means = center_mass(masks)
        
        masks = [
            maskUtils.encode(np.asfortranarray(mask))
            for mask in masks.astype(np.uint8)
        ]
        
        if len(self.active_tracks) == 0:
            
            for i in range(n):
                self.active_tracks.append(
                    TrackElement(
                        t=frame_count,
                        id=i,
                        mask=masks[i],
                        track_id=self.next_inst_id,
                        embed=embeds[i],
                        mean=means[i],
                    )
                )
                result[i] = self.next_inst_id
                
                self.next_inst_id += 1

            return result
        
        # compare inst by inst.
        # only compare with previous embeds, not including embeds of this frame
        last_reids = np.concatenate(
            [
                el.embed[np.newaxis]
                for el in self.active_tracks
            ], axis=0,
        )
        
        # cost matrix
        asso_sim = np.zeros(
            shape=(n, len(self.active_tracks)),
            dtype=np.float32,
        )
        
        # array for assigned instances
        detections_unassigned = np.ones(
            shape=(n, ),
            dtype=bool,
        )
        
        # step 1. Use distance between embeddings
        asso_sim += self.euclidean_scale * (
            self.euclidean_offset - cdist(embeds, last_reids)
        )
        
        # step 2. Use distance between masks (IoU)
        if self.use_mask_iou:
            asso_sim += self.iou_scale * maskUtils.iou(
                masks,
                [v.mask for v in self.active_tracks],
                np.zeros(len(self.active_tracks)),
            )
        
        cost_mat = asso_sim.max() - asso_sim
        cost_mat[asso_sim <= self.association_threshold] = 1e9
        
        rows, cols = linear_sum_assignment(cost_mat, maximize=False)

        for row, column in zip(rows, cols):
        
            # the instance was the same
            if euclidean(self.active_tracks[column].mean, means[row]) < self.means_threshold \
             and cost_mat[row][column] != 1e9:
                current_inst = TrackElement(
                    t=frame_count,
                    id=row,
                    mask=masks[row],
                    track_id=self.active_tracks[column].track_id,
                    embed=embeds[row],
                    mean=means[row],
                )
                self.active_tracks[column] = current_inst

                result[row] = self.active_tracks[column].track_id
            
            # it is a new instance
            else:
                current_inst = TrackElement(
                    t=frame_count,
                    id=row,
                    mask=masks[row],
                    track_id=self.next_inst_id,
                    embed=embeds[row],
                    mean=means[row],
                )
                self.active_tracks.append(current_inst)

                result[row] = self.next_inst_id
                self.next_inst_id += 1

            detections_unassigned[row] = False

        # new track id for unassigned instances
        for i in detections_unassigned.nonzero()[0]:
            
            current_inst = TrackElement(
                t=frame_count,
                id=i,
                mask=masks[i],
                track_id=self.next_inst_id,
                embed=embeds[i],
                mean=means[i],
            )
            
            self.active_tracks.append(current_inst)
            result[i] = self.next_inst_id
            self.next_inst_id += 1

        return result
