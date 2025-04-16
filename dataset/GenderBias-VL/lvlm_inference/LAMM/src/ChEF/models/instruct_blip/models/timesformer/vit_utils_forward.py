def forward(self, x):
    return drop_path(x, self.drop_prob, self.training)
