def __init__(self, cfg, model, tta_mapper=None, batch_size=3):
    """
        Args:
            cfg (CfgNode):
            model (GeneralizedRCNN): a GeneralizedRCNN to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
    super().__init__()
    if isinstance(model, DistributedDataParallel):
        model = model.module
    assert isinstance(model, GeneralizedRCNN
        ), 'TTA is only supported on GeneralizedRCNN. Got a model of type {}'.format(
        type(model))
    self.cfg = cfg.clone()
    assert not self.cfg.MODEL.KEYPOINT_ON, 'TTA for keypoint is not supported yet'
    assert not self.cfg.MODEL.LOAD_PROPOSALS, 'TTA for pre-computed proposals is not supported yet'
    self.model = model
    if tta_mapper is None:
        tta_mapper = DatasetMapperTTA(cfg)
    self.tta_mapper = tta_mapper
    self.batch_size = batch_size
