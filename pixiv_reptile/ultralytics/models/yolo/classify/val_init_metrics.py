def init_metrics(self, model):
    """Initialize confusion matrix, class names, and top-1 and top-5 accuracy."""
    self.names = model.names
    self.nc = len(model.names)
    self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf,
        task='classify')
    self.pred = []
    self.targets = []
