def get_intermediate_layers(self, x: torch.Tensor, n: Union[int, Sequence]=
    1, reshape: bool=False, return_prefix_tokens: bool=False, norm: bool=False
    ) ->Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
    """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
    outputs = self._intermediate_layers(x, n)
    if norm:
        outputs = [self.norm(out) for out in outputs]
    prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
    outputs = [out[:, self.num_prefix_tokens:] for out in outputs]
    if reshape:
        grid_size = self.patch_embed.grid_size
        outputs = [out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).
            permute(0, 3, 1, 2).contiguous() for out in outputs]
    if return_prefix_tokens:
        return tuple(zip(outputs, prefix_tokens))
    return tuple(outputs)
