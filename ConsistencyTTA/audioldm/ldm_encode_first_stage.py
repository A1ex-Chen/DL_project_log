@torch.no_grad()
def encode_first_stage(self, x):
    return self.first_stage_model.encode(x)
