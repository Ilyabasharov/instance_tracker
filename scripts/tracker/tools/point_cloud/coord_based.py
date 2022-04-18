import torch
from typing import Tuple


class PointCloudSampler(torch.nn.Module):

    def __init__(
        self,
        bg_num: int,
        fg_num: int,
        cat_emb: list,
        device: str,
        expand_ratio: float=0.2,
        offset_max: float=128.,
        *args,
        **kwargs,
    ) -> None:

        super().__init__()

        self.bg_num = bg_num
        self.fg_num = fg_num
        
        self.cat_emb = torch.tensor(
            data=cat_emb,
            dtype=torch.float32,
            device=device,
        )

        self.expand_ratio = expand_ratio
        self.offset_max = offset_max

        self.device = device

    @torch.jit.export
    def find_expanded_coords(
        self,
        minMaxU: Tuple[int, int],
        minMaxV: Tuple[int, int],
        dims: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:

        vMax, uMax = dims

        u0, u1 = minMaxU
        v0, v1 = minMaxV

        vlen = v1 - v0
        ulen = u1 - u0

        # enlarge bbox
        v0 = max(0, v0 - int(self.expand_ratio * vlen))
        v1 = min(v1 + int(self.expand_ratio * vlen), vMax - 1)
        u0 = max(0, u0 - int(self.expand_ratio * ulen))
        u1 = min(u1 + int(self.expand_ratio * ulen), uMax - 1)
        
        return (v0, v1), (u0, u1)

    @torch.jit.export
    def forward(
        self,
        image: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple:

        #original dimensions of image
        n, vMax, uMax = masks.shape

        #allocated memory for object positions
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

        #compute objects field
        field = masks.sum(
            dim=0,
            dtype=torch.bool,
        ).int().mul_(2)

        for i in range(n):

            mask = masks[i]
        
            #indexes where mask has true indexes
            vu = mask.nonzero()
            
            vs = vu[:, 0]
            us = vu[:, 1]

            #relative bbox coordinates
            v0, v1 = vs.min(), vs.max()
            u0, u1 = us.min(), us.max()

            xyxys[i, 0] = u0
            xyxys[i, 1] = v0
            xyxys[i, 2] = u1
            xyxys[i, 3] = v1

            (v0, v1), (u0, u1) = self.find_expanded_coords(
                minMaxU=(u0.item(), u1.item() + 1),
                minMaxV=(v0.item(), v1.item() + 1),
                dims=(vMax, uMax),
            )
            
            field[mask] = 1

            crop_mask = mask[v0: v1, u0: u1]
            crop_image = image[v0: v1, u0: u1]
            crop_field = field[v0: v1, u0: u1]

            #step 1. foreground
            #expanded mask
            vu = crop_mask.nonzero()
            
            vs = vu[:, 0]
            us = vu[:, 1]
            
            vs, us = vs.float(), us.float()
            vc, uc = vs.mean(), us.mean()

            vs.sub_(vc).div_(self.offset_max).unsqueeze_(-1)
            us.sub_(uc).div_(self.offset_max).unsqueeze_(-1)

            #shape (crop_mask.count_nonzero(), 5)
            pointUVs = torch.cat(
                tensors=[
                    crop_image[crop_mask], # (crop_mask.count_nonzero(), 3)
                    vs,                    # (crop_mask.count_nonzero(), 1)
                    us,                    # (crop_mask.count_nonzero(), 1)
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
                    dtype=torch.long,
                )
            ]

            #step 2. background
            crop_mask = torch.bitwise_not(
                crop_mask,
            )

            # if mask not empty
            if crop_mask.count_nonzero() != 0:

                vu = crop_mask.nonzero()
                
                vs = vu[:, 0]
                us = vu[:, 1]

                vs = vs.float().sub_(vc).div_(self.offset_max).unsqueeze_(-1)
                us = us.float().sub_(uc).div_(self.offset_max).unsqueeze_(-1)

                pointUVs = torch.cat(
                    tensors=[
                        crop_image[crop_mask],
                        vs,
                        us, 
                        torch.index_select(
                            self.cat_emb,
                            dim=0,
                            index=crop_field[crop_mask],
                        ),
                    ],
                    dim=1,
                )

                #result of 2 step: random sampled background pointcloud
                #shape: (bg_num, 8)
                points[i, self.fg_num:, ] = pointUVs[
                    torch.randint(
                        low=0,
                        high=pointUVs.shape[0],
                        size=(self.bg_num, ),
                        device=self.device,
                        dtype=torch.long,
                    )
                ]

            field[mask] = 2

        xyxys[..., 0].div_(uMax)
        xyxys[..., 1].div_(vMax)
        xyxys[..., 2].div_(uMax)
        xyxys[..., 3].div_(vMax)

        return points, xyxys
