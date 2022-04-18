from __future__ import annotations

import torch
from tracker.tools.assign.assigner import Assigner

from tracker.tools.filter import FilterDetections
from tracker.tools.assign.assigner import Assigner
from tracker.tools.point_cloud.coord_based import PointCloudSampler

from tracker.pointtrack.models.utils import get_embedding_model


class PointTrack:

    def __init__(
        self,
        params: dict,
    ) -> PointTrack:

        self.embedder = dict()
        self.assigner = dict()

        for classname in params['classes']:

            self.embedder[classname] = get_embedding_model(
                params=params['embedder'],
                classname=classname,
            )

            self.assigner[classname] = Assigner(
                **params['assigner'][classname],
            )

        # define sampler
        sampler = PointCloudSampler(
            **params['sampler'],
        )

        if params['sampler']['jit']:
            self.sampler = torch.jit.script(sampler)
        else:
            self.sampler = sampler

        # define filter
        filter = FilterDetections(
            **params['filter'],
        )

        if params['filter']['jit']:
            self.filter = torch.jit.script(filter)
        else:
            self.filter = filter

        self.frame_count = -1
        self.classes = params['classes']

        #preprocess params
        self.mask_scale_factor = params['preprocess']['mask_scale_factor']
        self.image_devide_factor = params['preprocess']['image_devide_factor']

    @torch.no_grad()
    def inference(
        self,
        image: torch.Tensor,
        masks: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
    ) -> list:

        '''
        Main stage of tracker
        '''

        result = []

        self.frame_count += 1

        filtering_mask = self.filter.main_stage(
            labels=labels,
            scores=scores,
            masks=masks,
        )

        # prevent torch2trt error
        idx = filtering_mask.cpu().numpy()

        masks = masks[idx]
        boxes = boxes[idx]
        labels = labels[idx]
        scores = scores[idx]

        N = len(masks)

        track_ids = torch.zeros(
            size=(N, ),
            dtype=torch.int32,
            device=masks.device,
        )

        # if there are no objects
        if N == 0:
            return labels, masks, track_ids

        resized_masks, points, xyxys = self.preprocess_input(
            image=image,
            masks=masks,
        )

        for classname in self.classes:

            class_filtering_mask = self.filter.per_class(
                labels=labels,
                classname=classname,
            )

            if class_filtering_mask.count_nonzero() == 0:
                continue

            # prevent torch2trt error
            idx = class_filtering_mask.cpu().numpy()

            class_indexes, = class_filtering_mask.nonzero(
                as_tuple=True,
            )

            embeds = self.embedder[classname](
                points[idx],
                xyxys[idx],
            )

            tracked_indexes = self.assigner[classname].tracking(
                frame_count=self.frame_count,
                embeds=embeds.cpu().numpy(),
                masks=resized_masks[idx].cpu().numpy(),
            )

            track_ids.scatter_(
                dim=0,
                index=class_indexes,
                src=torch.from_numpy(tracked_indexes).to(class_indexes.device),
            )

        return labels, masks, track_ids

    def preprocess_input(
        self,
        image: torch.tensor,
        masks: torch.tensor,
        *args,
        **kwargs,
    ) -> tuple:

        '''
        Takes masks (NxHxW), image (3xHxW)
        and returns result_mask, resized_masks
        and point_clouds to increase processing speed
        '''

        resized_masks = masks
        preprocessed_image = image.div(self.image_devide_factor)

        if self.mask_scale_factor != 1:

            resized_masks = torch.nn.functional.interpolate(
                input=masks.unsqueeze(1),             # (NxHxW) -> (Nx1xHxW)
                scale_factor=self.mask_scale_factor,
                mode='nearest',
                recompute_scale_factor=True,
            ).squeeze(1)                              # (Nx1xHxW) -> (NxHxW)

        points, xyxys = self.sampler(
            image=preprocessed_image,
            masks=masks.bool(),
        )

        return resized_masks, points, xyxys