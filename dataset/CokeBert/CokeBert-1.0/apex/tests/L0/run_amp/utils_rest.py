import torch

HALF = 'torch.cuda.HalfTensor'
FLOAT = 'torch.cuda.FloatTensor'

DTYPES = [torch.half, torch.float]

ALWAYS_HALF = {torch.float: HALF,
               torch.half: HALF}
ALWAYS_FLOAT = {torch.float: FLOAT,
                torch.half: FLOAT}
MATCH_INPUT = {torch.float: FLOAT,
               torch.half: HALF}
