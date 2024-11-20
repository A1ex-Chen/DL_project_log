@torch.jit.ignore
def set_grad_checkpointing(self, enable=True):
    self.visual.set_grad_checkpointing(enable)
