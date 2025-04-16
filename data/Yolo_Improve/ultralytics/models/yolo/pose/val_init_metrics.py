def init_metrics(self, model):
    """Initiate pose estimation metrics for YOLO model."""
    super().init_metrics(model)
    self.kpt_shape = self.data['kpt_shape']
    is_pose = self.kpt_shape == [17, 3]
    nkpt = self.kpt_shape[0]
    self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt
    self.stats = dict(tp_p=[], tp=[], conf=[], pred_cls=[], target_cls=[],
        target_img=[])
