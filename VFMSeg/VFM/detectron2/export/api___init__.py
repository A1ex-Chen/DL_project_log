def __init__(self, predict_net, init_net):
    super().__init__()
    self.eval()
    self._predict_net = predict_net
    self._init_net = init_net
    self._predictor = None
