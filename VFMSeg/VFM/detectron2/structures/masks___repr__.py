@torch.jit.unused
def __repr__(self) ->str:
    s = self.__class__.__name__ + '('
    s += 'num_instances={})'.format(len(self.tensor))
    return s
