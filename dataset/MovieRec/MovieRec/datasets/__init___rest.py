from .ml_1m import ML1MDataset
from .ml_20m import ML20MDataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    ML20MDataset.code(): ML20MDataset
}

