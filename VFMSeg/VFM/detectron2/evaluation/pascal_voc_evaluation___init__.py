def __init__(self, dataset_name):
    """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
    self._dataset_name = dataset_name
    meta = MetadataCatalog.get(dataset_name)
    annotation_dir_local = PathManager.get_local_path(os.path.join(meta.
        dirname, 'Annotations/'))
    self._anno_file_template = os.path.join(annotation_dir_local, '{}.xml')
    self._image_set_path = os.path.join(meta.dirname, 'ImageSets', 'Main', 
        meta.split + '.txt')
    self._class_names = meta.thing_classes
    assert meta.year in [2007, 2012], meta.year
    self._is_2007 = meta.year == 2007
    self._cpu_device = torch.device('cpu')
    self._logger = logging.getLogger(__name__)
