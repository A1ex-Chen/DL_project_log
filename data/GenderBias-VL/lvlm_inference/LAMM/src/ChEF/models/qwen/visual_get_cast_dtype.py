def get_cast_dtype(self) ->torch.dtype:
    return self.resblocks[0].mlp.c_fc.weight.dtype
