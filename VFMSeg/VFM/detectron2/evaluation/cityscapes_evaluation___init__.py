def __init__(self, dataset_name):
    """
        Args:
            dataset_name (str): the name of the dataset.
                It must have the following metadata associated with it:
                "thing_classes", "gt_dir".
        """
    self._metadata = MetadataCatalog.get(dataset_name)
    self._cpu_device = torch.device('cpu')
    self._logger = logging.getLogger(__name__)
