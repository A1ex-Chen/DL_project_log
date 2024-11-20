def __init__(self, dataset_name, tasks=None, distributed=True, output_dir=
    None, *, max_dets_per_image=None):
    """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                "json_file": the path to the LVIS format annotation
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
            max_dets_per_image (None or int): limit on maximum detections per image in evaluating AP
                This limit, by default of the LVIS dataset, is 300.
        """
    from lvis import LVIS
    self._logger = logging.getLogger(__name__)
    if tasks is not None and isinstance(tasks, CfgNode):
        self._logger.warn(
            'COCO Evaluator instantiated using config, this is deprecated behavior. Please pass in explicit arguments instead.'
            )
        self._tasks = None
    else:
        self._tasks = tasks
    self._distributed = distributed
    self._output_dir = output_dir
    self._max_dets_per_image = max_dets_per_image
    self._cpu_device = torch.device('cpu')
    self._metadata = MetadataCatalog.get(dataset_name)
    json_file = PathManager.get_local_path(self._metadata.json_file)
    self._lvis_api = LVIS(json_file)
    self._do_evaluation = len(self._lvis_api.get_ann_ids()) > 0
