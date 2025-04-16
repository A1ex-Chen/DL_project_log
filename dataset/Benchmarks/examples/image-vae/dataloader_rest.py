import io

import cairosvg
import numpy as np
import torch
from invert import Invert
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from torchvision import transforms


class MoleLoader(torch.utils.data.Dataset):











