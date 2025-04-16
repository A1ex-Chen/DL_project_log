def __init__(self, filename, include_mask):
    """Constructs COCO evaluation class.

    The class provides the interface to metrics_fn in TPUEstimator. The
    _evaluate() loads a JSON file in COCO annotation format as the
    groundtruths and runs COCO evaluation.

    Args:
      filename: Ground truth JSON file name. If filename is None, use
        groundtruth data passed from the dataloader for evaluation.
      include_mask: boolean to indicate whether or not to include mask eval.
    """
    if filename:
        if filename.startswith('gs://'):
            _, local_val_json = tempfile.mkstemp(suffix='.json')
            tf.io.gfile.remove(local_val_json)
            tf.io.gfile.copy(filename, local_val_json)
            atexit.register(tf.io.gfile.remove, local_val_json)
        else:
            local_val_json = filename
        self.coco_gt = MaskCOCO(local_val_json)
    self.filename = filename
    self.metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
        'ARmax1', 'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']
    self._include_mask = include_mask
    if self._include_mask:
        mask_metric_names = [('mask_' + x) for x in self.metric_names]
        self.metric_names.extend(mask_metric_names)
    self._reset()
