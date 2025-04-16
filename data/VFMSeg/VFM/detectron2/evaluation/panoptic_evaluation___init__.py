def __init__(self, dataset_name: str, output_dir: Optional[str]=None):
    """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
    self._metadata = MetadataCatalog.get(dataset_name)
    self._thing_contiguous_id_to_dataset_id = {v: k for k, v in self.
        _metadata.thing_dataset_id_to_contiguous_id.items()}
    self._stuff_contiguous_id_to_dataset_id = {v: k for k, v in self.
        _metadata.stuff_dataset_id_to_contiguous_id.items()}
    self._output_dir = output_dir
    if self._output_dir is not None:
        PathManager.mkdirs(self._output_dir)
