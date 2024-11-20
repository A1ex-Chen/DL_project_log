def reset_parameters(self):
    nn.init.zeros_(self.fc_gamma.weight)
    nn.init.zeros_(self.fc_beta.weight)
    nn.init.ones_(self.fc_gamma.bias)
    nn.init.zeros_(self.fc_beta.bias)
