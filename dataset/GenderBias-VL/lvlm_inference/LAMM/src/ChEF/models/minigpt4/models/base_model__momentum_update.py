@torch.no_grad()
def _momentum_update(self):
    for model_pair in self.model_pairs:
        for param, param_m in zip(model_pair[0].parameters(), model_pair[1]
            .parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1.0 -
                self.momentum)
