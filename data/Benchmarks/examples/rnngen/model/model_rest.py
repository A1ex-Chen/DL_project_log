import torch.nn as nn
import torch.nn.functional as F


# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5836943/pdf/MINF-37-na.pdf
class CharRNN(nn.Module):

    # pass x as a pack padded sequence please.
