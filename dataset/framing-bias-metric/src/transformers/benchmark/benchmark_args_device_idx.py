@property
@torch_required
def device_idx(self) ->int:
    return torch.cuda.current_device()
