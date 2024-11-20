def get_cast_device(self) ->torch.device:
    return self.resblocks[0].mlp.c_fc.weight.device
