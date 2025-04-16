def init_metrics(self, model):
    """Initialize evaluation metrics for YOLO."""
    val = self.data.get(self.args.split, '')
    self.is_coco = isinstance(val, str) and 'coco' in val and val.endswith(
        f'{os.sep}val2017.txt')
    self.is_lvis = isinstance(val, str) and 'lvis' in val and not self.is_coco
    self.class_map = converter.coco80_to_coco91_class(
        ) if self.is_coco else list(range(len(model.names)))
    self.args.save_json |= (self.is_coco or self.is_lvis) and not self.training
    self.names = model.names
    self.nc = len(model.names)
    self.metrics.names = self.names
    self.metrics.plot = self.args.plots
    self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
    self.seen = 0
    self.jdict = []
    self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[]
        )
