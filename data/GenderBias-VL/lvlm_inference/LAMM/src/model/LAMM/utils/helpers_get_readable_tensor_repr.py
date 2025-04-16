@staticmethod
def get_readable_tensor_repr(name: str, tensor: torch.Tensor) ->str:
    st = '(' + name + '): ' + 'tensor(' + str(tuple(tensor[1].shape)
        ) + ', requires_grad=' + str(tensor[1].requires_grad) + ')\n'
    return st
