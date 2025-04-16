import torch
from pathlib import Path
import os








if __name__ == "__main__":

    p = torch.tensor([0.5, 0.6, 0.7]).view(3, 1)
    p_ = 1 - p
    p = torch.cat([p, p_], dim=1).view(-1, 2)
    print(p)
    q = torch.tensor([0.5, 0.6, 0.7]).view(3, 1)
    q_ = 1 - q
    q = torch.cat([q, q_], dim=1).view(-1, 2)
    print(q.shape)
    kl = torch.nn.functional.kl_div(torch.log(q), p, reduction="sum")

    print(kl)