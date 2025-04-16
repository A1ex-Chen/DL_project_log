from abc import abstractmethod

import pandas as pd


class Dataset:
    """Abstract dataset - Used for both Keras and Pytorch"""

    @abstractmethod

    @abstractmethod




class InMemoryDataset(Dataset):
    """Abstract class for in memory data"""






class Subset(InMemoryDataset):
    """Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The dataset to be subsetted
        indices (sequence): Indices in the whole set selected for subset
    """



