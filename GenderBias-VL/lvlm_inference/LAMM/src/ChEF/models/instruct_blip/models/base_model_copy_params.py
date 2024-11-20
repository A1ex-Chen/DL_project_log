@torch.no_grad()
def copy_params(self):
    for model_pair in self.model_pairs:
        for param, param_m in zip(model_pair[0].parameters(), model_pair[1]
            .parameters()):
            param_m.data.copy_(param.data)
            param_m.requires_grad = False
