import os
import argparse
import torch
import soundfile as sf

from tools.torch_tools import seed_all
from audioldm_eval import EvaluationHelper


device = torch.device(
    "cuda:0" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)






if __name__ == "__main__":
    main()