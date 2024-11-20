def __init__(self, prediction, joint_pred, min_lstm_bs):
    super(RNNTPredict, self).__init__()
    self.prediction = prediction
    self.joint_pred = joint_pred
    self.min_lstm_bs = min_lstm_bs
