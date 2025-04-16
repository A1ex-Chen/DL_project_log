def __init__(self, dataset_name, tasks=None, distributed=True, output_dir=
    None, *, max_dets_per_image=None, use_fast_impl=True, kpt_oks_sigmas=(),
    allow_cached_coco=True, force_tasks=None, refcoco=False):
    """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                   contains all the results in the format they are produced by the model.
                2. "coco_instances_results.json" a json file in COCO's result format.
            max_dets_per_image (int): limit on the maximum number of detections per image.
                By default in COCO, this limit is to 100, but this can be customized
                to be greater, as is needed in evaluation metrics AP fixed and AP pool
                (see https://arxiv.org/pdf/2102.01066.pdf)
                This doesn't affect keypoint evaluation.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
            kpt_oks_sigmas (list[float]): The sigmas used to calculate keypoint OKS.
                See http://cocodataset.org/#keypoints-eval
                When empty, it will use the defaults in COCO.
                Otherwise it should be the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
            allow_cached_coco (bool): Whether to use cached coco json from previous validation
                runs. You should set this to False if you need to use different validation data.
                Defaults to True.
        """
    self.dataset_name = dataset_name
    self._logger = logging.getLogger(__name__)
    self._distributed = distributed
    self._output_dir = output_dir
    self.force_tasks = force_tasks
    self.refcoco = refcoco
    if use_fast_impl and COCOeval_opt is COCOeval:
        self._logger.info(
            'Fast COCO eval is not built. Falling back to official COCO eval.')
        use_fast_impl = False
    self._use_fast_impl = use_fast_impl
    if max_dets_per_image is None:
        max_dets_per_image = [1, 10, 100]
    else:
        max_dets_per_image = [1, 10, max_dets_per_image]
    self._max_dets_per_image = max_dets_per_image
    if tasks is not None and isinstance(tasks, CfgNode):
        kpt_oks_sigmas = (tasks.TEST.KEYPOINT_OKS_SIGMAS if not
            kpt_oks_sigmas else kpt_oks_sigmas)
        self._logger.warn(
            'COCO Evaluator instantiated using config, this is deprecated behavior. Please pass in explicit arguments instead.'
            )
        self._tasks = None
    else:
        self._tasks = tasks
    self._cpu_device = torch.device('cpu')
    self._metadata = MetadataCatalog.get(dataset_name)
    if not hasattr(self._metadata, 'json_file'):
        if output_dir is None:
            raise ValueError(
                'output_dir must be provided to COCOEvaluator for datasets not in COCO format.'
                )
        self._logger.info(
            f"Trying to convert '{dataset_name}' to COCO format ...")
        cache_path = os.path.join(output_dir,
            f'{dataset_name}_coco_format.json')
        self._metadata.json_file = cache_path
        convert_to_coco_json(dataset_name, cache_path, allow_cached=
            allow_cached_coco)
    json_file = PathManager.get_local_path(self._metadata.json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        self._coco_api = COCO(json_file)
    self._do_evaluation = 'annotations' in self._coco_api.dataset
    if self._do_evaluation:
        self._kpt_oks_sigmas = kpt_oks_sigmas
