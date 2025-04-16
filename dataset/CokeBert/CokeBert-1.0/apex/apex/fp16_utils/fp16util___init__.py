def __init__(self, network):
    super(FP16Model, self).__init__()
    self.network = convert_network(network, dtype=torch.half)
