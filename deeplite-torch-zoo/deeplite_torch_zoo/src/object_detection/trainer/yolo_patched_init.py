def patched_init(obj, model_name=None, torch_model=None, num_classes=None,
    task=None, session=None, pretrained=False, pretraining_dataset='coco'):
    if isinstance(obj, nn.Module):
        nn.Module.__init__(obj)
    if model_name is None and torch_model is None:
        raise ValueError(
            'Either a `model_name` string or a `torch_model` (nn.Module object) must be passed to instantiate a trainer object.'
            )
    obj.callbacks = callbacks.get_default_callbacks()
    obj.predictor = None
    obj.model = None
    obj.trainer = None
    obj.task = task if task is not None else 'detect'
    obj.ckpt = True
    obj.cfg = None
    obj.ckpt_path = None
    obj.overrides = {}
    obj.metrics = None
    obj.session = session
    obj.num_classes = num_classes
    obj.cfg = model_name
    if model_name == 'yolov8n.pt':
        model_name = 'yolo8n'
    if model_name is not None:
        obj.model = get_model(model_name=model_name, dataset_name=
            pretraining_dataset, pretrained=pretrained, num_classes=
            num_classes, custom_head='yolo8')
    else:
        obj.model = torch_model
    obj.model._forward = obj.model.forward

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self._forward(x, *args, **kwargs)

    def loss(self, batch, preds=None):
        if not hasattr(self, 'criterion'):
            self.criterion = v8DetectionLoss(self)
        preds = self._forward(batch['img']) if preds is None else preds
        return self.criterion(preds, batch)
    obj.model.loss = types.MethodType(loss, obj.model)
    obj.model.forward = types.MethodType(forward, obj.model)
    obj.model.names = [''] if not num_classes else [f'class{i}' for i in
        range(num_classes)]
    obj.overrides['model'] = obj.cfg
    args = {**DEFAULT_CFG_DICT, **obj.overrides}
    obj.model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}
    obj.model.task = obj.task
    obj.model.model_name = model_name
