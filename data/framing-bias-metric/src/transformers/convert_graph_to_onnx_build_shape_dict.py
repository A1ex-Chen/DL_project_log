def build_shape_dict(name: str, tensor, is_input: bool, seq_len: int):
    if isinstance(tensor, (tuple, list)):
        return [build_shape_dict(name, t, is_input, seq_len) for t in tensor]
    else:
        axes = {[axis for axis, numel in enumerate(tensor.shape) if numel ==
            1][0]: 'batch'}
        if is_input:
            if len(tensor.shape) == 2:
                axes[1] = 'sequence'
            else:
                raise ValueError(
                    f'Unable to infer tensor axes ({len(tensor.shape)})')
        else:
            seq_axes = [dim for dim, shape in enumerate(tensor.shape) if 
                shape == seq_len]
            axes.update({dim: 'sequence' for dim in seq_axes})
    print(
        f"Found {'input' if is_input else 'output'} {name} with shape: {axes}")
    return axes
