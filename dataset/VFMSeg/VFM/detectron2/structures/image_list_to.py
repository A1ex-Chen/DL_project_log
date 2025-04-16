@torch.jit.unused
def to(self, *args: Any, **kwargs: Any) ->'ImageList':
    cast_tensor = self.tensor.to(*args, **kwargs)
    return ImageList(cast_tensor, self.image_sizes)
