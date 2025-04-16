def forward(self, s0, s1, weights):
    """
        :param s0:
        :param s1:
        :param weights: [14, 8]
        :return:
        """
    states = [s0, s1]
    offset = 0
    for i in range(self.num_nodes):
        offset += len(states)
    return torch.cat(states[-self.multiplier:], dim=1)
