import torch
import torch.nn as nn


class TrackingRankingLoss(nn.Module):

    def __init__(
        self,
        margin: float,
    ) -> None:

        super().__init__()

        self.ranking_loss = nn.MarginRankingLoss(
            margin=margin,
            reduction='mean',
        )

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2,)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        loss = torch.zeros([1], device=inputs.device)
        if mask.float().unique().shape[0] > 1:
            dist_ap, dist_an = [], []
            for i in range(n):
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            dist_ap = torch.cat(dist_ap)
            dist_an = torch.cat(dist_an)
            # Compute ranking hinge loss
            y = torch.ones_like(dist_an)
            loss = self.ranking_loss(dist_an, dist_ap, y).unsqueeze(0)

        return loss