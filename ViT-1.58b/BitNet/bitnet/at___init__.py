def __init__(self, net, max_seq_len=2048, pad_value=0):
    super().__init__()
    self.max_seq_len = max_seq_len
    self.pad_value = pad_value
    self.net = net
