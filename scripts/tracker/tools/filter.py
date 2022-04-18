import torch
import utils
import numpy as np


class FilterDetections(torch.nn.Module):

    def __init__(
        self,
        n_max_objects: int,
        threshold_score: float,
        threshold_area: float,
        class_mapping: dict,
        device: str,
        *args,
        **kwargs,
    ) -> None:
        
        super().__init__()

        self.n_max_objects = n_max_objects
        self.threshold_score = threshold_score
        self.threshold_area = threshold_area
        
        all_classes, per_class = [], dict()

        for classname in class_mapping:

            all_classes += class_mapping[classname]

            per_class[classname] = torch.tensor(
                data=class_mapping[classname],
                device=device,
            )

        self.all_classes = torch.tensor(
            data=all_classes,
            device=device,
        )

        self.class_mapping = per_class

    @torch.jit.ignore
    def main_stage(
        self,
        labels: torch.Tensor,
        scores: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:

        '''
        filter unnecessary objects
        '''

        if len(scores) * len(labels) * len(masks) == 0:

            result_mask = torch.tensor(
                data=[],
                dtype=torch.bool,
                device=masks.device,
            )

            return result_mask

        result_mask = self.last_steps(
            masks=masks,
            result_mask=self.first_steps(
                labels=labels,
                scores=scores,
            )
        )

        return result_mask

    @torch.jit.export
    def first_steps(
        self,
        labels: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
            
        # 1st step: by score
        score_mask = scores > self.threshold_score

        #2st step: by top k 
        top_k_mask = utils.topk_torch(
            array=scores,
            k=self.n_max_objects,
        )

        #3st step: by labels
        labels_mask = torch.isin(
            elements=labels,
            test_elements=self.all_classes,
            assume_unique=False,
            invert=False,
        )
        
        result_mask = \
            score_mask & \
            labels_mask & \
            top_k_mask

        return result_mask

    @torch.jit.ignore
    def last_steps(
        self,
        masks: torch.Tensor,
        result_mask: torch.Tensor,
    ) -> torch.Tensor:

        #4th step: by area
        N, H, W = masks.shape

        result_mask_cpu = result_mask.cpu().numpy()

        area_mask = torch.count_nonzero(
            masks[result_mask_cpu],
            dim=(1, 2),
        ) > self.threshold_area * H * W

        result_mask[result_mask_cpu] = area_mask

        return result_mask
    
    @torch.jit.export
    def per_class(
        self,
        labels: torch.Tensor,
        classname: str,
    ) -> torch.Tensor:

        result_mask = torch.isin(
            elements=labels,
            test_elements=self.class_mapping[classname],
            assume_unique=False,
            invert=False,
        )

        return result_mask
