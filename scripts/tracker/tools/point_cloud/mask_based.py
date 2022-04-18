#TODO
from __future__ import annotations

import torch
import kornia

class PointCloudSampler:

    def __init__(
        self,
        cat_emb: torch.tensor,
        device: torch.device,
        offset_max: float=128.,
        expand_ratio: float=0.04,
        *args,
        **kwargs,
    ) -> None:

        self.cat_emb = torch.tensor(
            data=cat_emb,
            dtype=torch.float32,
            device=device,
        )

        self.offset_max = offset_max
        self.expand_ratio = expand_ratio

        self.device = device

    def sample(
        self,
        image: torch.tensor, # (H, W, 3) dtype
        masks: torch.tensor, # (N, 1, H, W)
    ) -> tuple:

        #original dimensions of image
        n, vMax, uMax = masks.shape

        xyxys = torch.zeros(
            (n, 4),
            dtype=torch.float32,
            device=self.device,
        )

        #allocated memory for pointcloud
        points = torch.zeros(
            (n, self.bg_num + self.fg_num, 8),
            dtype=torch.float32,
            device=self.device,
        )

        #expanded dims
        dv = vMax * self.expand_ratio
        du = uMax * self.expand_ratio

        #difference
        background_masks = kornia.morphology.dilation(
            tensor=masks,
            kernel=torch.ones(
                size=(dv, du),
                device=self.device,
                dtype=torch.uint8,
            ),
            max_val=0,
            engine='unfold',
        ) - masks

        #compute objects field
        field = masks.sum(
            dim=0,
            dtype=torch.bool,
        ).int().mul_(2)

        for i in range(n):

            #indexes where mask has true indexes
            vs, us = masks[i].nonzero(as_tuple=True)

            vs, us = vs.float(), us.float()
            vc, uc = vs.mean(), us.mean()

            #bbox coordinates
            xyxys[i, 0] = us.min()
            xyxys[i, 1] = vs.min()
            xyxys[i, 2] = us.max()
            xyxys[i, 3] = vs.max()

            vs.sub_(vc).div_(self.offset_max).unsqueeze_(-1)
            us.sub_(uc).div_(self.offset_max).unsqueeze_(-1)

            #shape (crop_mask.count_nonzero(), 5)
            pointUVs = torch.cat(
                tensors=[
                    image[masks[i]], #masks[i].count_nonzero() x 3
                    vs,              #cx
                    us,              #cy
                ],
                dim=1,
            )

            #result of 1 step: random sampled foreground pointcloud
            #shape: (fg_num, 5)
            points[i, :self.fg_num, :5] = pointUVs[
                torch.randint(
                    low=0,
                    high=pointUVs.shape[0],
                    size=(self.fg_num, ),
                    device=self.device,
                )
            ]

            #step 2. background
            vs, us = background_masks[i].nonzero(as_tuple=True)

            vs, us = vs.float(), us.float()
            vc, uc = vs.mean(), us.mean()

            vs.sub_(vc).div_(self.offset_max).unsqueeze_(-1)
            us.sub_(uc).div_(self.offset_max).unsqueeze_(-1)
            
            field[masks[i]] = 1

            pointUVs = torch.cat(
                tensors=[
                    image[background_masks[i]],
                    vs,
                    us,
                    torch.index_select(
                        input=self.cat_emb,
                        dim=0,
                        index=field[background_masks[i]],
                    ),
                ],
                dim=1,
            )

            field[masks[i]] = 2

            #result of 2 step: random sampled background pointcloud
            #shape: (bg_num, 8)
            points[i, self.fg_num:, ] = pointUVs[
                torch.randint(
                    low=0,
                    high=pointUVs.shape[0],
                    size=(self.bg_num, ),
                    device=self.device,
                )
            ]

        xyxys[..., 0].div_(uMax)
        xyxys[..., 1].div_(vMax)
        xyxys[..., 2].div_(uMax)
        xyxys[..., 3].div_(vMax)

        return points, xyxys




