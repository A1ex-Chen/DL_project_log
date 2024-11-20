@staticmethod
def get_tainable_param(trainable_param: torch.Tensor):
    return torch.nn.Parameter(trainable_param).requires_grad_(requires_grad
        =True)
