def __handle_values__(self, values, indices: Union[int, float, List[Union[
    int, float]], torch.Tensor, slice]):
    if isinstance(indices, int) or isinstance(indices, float):
        return [values]
    return values
