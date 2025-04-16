def forward(self, s0, s1, weights):
    """
        :param s0:
        :param s1:
        :param weights: [14, 8]
        :return:
        """
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    states = [s0, s1]
    offset = 0
    for i in range(self.num_nodes):
        s = sum(self.layers[offset + j](h, weights[offset + j]) for j, h in
            enumerate(states))
        offset += len(states)
        states.append(s)
    return torch.cat(states[-self.multiplier:], dim=1)
