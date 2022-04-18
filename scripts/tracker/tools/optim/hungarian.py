import torch

__docformat__ = 'markdown'
__all__       = ['compute', ]
__version__   = '1.0.0'
__author__    = 'Ilya Basharov, ilya.basharov.98@mail.ru'
__license__   = 'MIT License'


def compute(
    X: torch.Tensor,
    eps: float=1e-4,
) -> tuple:
    """
        X: n-by-n matrix w/ integer entries
        eps: "bid size" -- smaller values means higher accuracy w/ longer runtime
    """

    eps = 1 / X.shape[0] if eps is None else eps
    
    if X.shape[1] == 1:
        
        return (
            X.T.argmin().unsqueeze(0),
            torch.tensor(
                [0, ],
                dtype=torch.long,
                device=X.device,
            )
        )
    
    # --
    # Init
    
    cost     = torch.zeros(
        size=(1, X.shape[1]),
        device=X.device,
        dtype=torch.float32,
    )
    curr_ass = torch.full(
        size=(X.shape[0], ),
        fill_value=-1.,
        device=X.device,
        dtype=torch.int64,
    )
    bids     = torch.zeros(
        size=X.shape,
        device=X.device,
        dtype=torch.float32,
    )
    
    counter = 0
    
    while (curr_ass == -1).any():

        counter += 1
        
        # --
        # Bidding
        
        unassigned = (curr_ass == -1).nonzero().squeeze()

        if len(unassigned.shape) < 1:
            unassigned.unsqueeze_(dim=0)
        
        value = X[unassigned] - cost
        top_value, top_idx = value.topk(2, dim=1)
        
        first_idx = top_idx[:, 0]
        first_value, second_value = top_value[:, 0], top_value[:, 1]
        
        bid_increments = first_value - second_value + eps
        
        bids_ = bids[unassigned]
        bids_.zero_()

        if len(bids_.shape) < 2:
            bids_.unsqueeze_(dim=0)

        bids_.scatter_(
            dim=1,
            index=first_idx.contiguous().view(-1, 1),
            src=bid_increments.view(-1, 1).float()
        )
        
        # --
        # Assignment
        
        have_bidder = (bids_ > 0).int().sum(dim=0).nonzero()
        
        high_bids, high_bidders = bids_[:, have_bidder].max(dim=0)
        high_bidders = unassigned[high_bidders.squeeze()]
        
        cost[:, have_bidder] += high_bids
        
        curr_ass[(curr_ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1)] = -1
        curr_ass[high_bidders] = have_bidder.squeeze()
    
    return (
        torch.arange(
            len(curr_ass),
            device=curr_ass.device,
            dtype=torch.int64,
        ),
        curr_ass,
    )
