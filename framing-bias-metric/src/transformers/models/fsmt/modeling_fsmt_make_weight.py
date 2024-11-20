def make_weight(self, num_positions, embedding_dim, padding_idx):
    weight = self.get_embedding(num_positions, embedding_dim, padding_idx)
    if not hasattr(self, 'weight'):
        super().__init__(num_positions, embedding_dim, padding_idx, _weight
            =weight)
    else:
        weight = weight.to(self.weight.device)
        self.weight = nn.Parameter(weight)
    self.weight.detach_()
    self.weight.requires_grad = False
