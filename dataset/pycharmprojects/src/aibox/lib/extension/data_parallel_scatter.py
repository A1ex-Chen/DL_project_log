def scatter(self, inputs: Tuple, kwargs: Dict, device_ids: List[int]) ->Tuple[
    List, List]:

    def scatter_map(obj):
        if isinstance(obj, Bunch):
            num_devices = len(self.device_ids)
            elements = obj
            num_elements = len(elements)
            num_chunks = num_devices
            chunk_sizes = [num_elements // num_chunks] * num_chunks
            for i in range(num_elements % num_chunks):
                chunk_sizes[i] += 1
            slice_ends = torch.tensor(chunk_sizes).cumsum(dim=0).tolist()
            slice_starts = [0] + slice_ends[:-1]
            elements_chunks = [elements[i:j] for i, j in zip(slice_starts,
                slice_ends)]
            elements_chunks = [Bunch([element.to(torch.device('cuda', self.
                device_ids[i])) for element in elements]) for i, elements in
                enumerate(elements_chunks)]
            scattered_obj = tuple(elements for elements in elements_chunks if
                elements)
            return scattered_obj
        if isinstance(obj, tuple):
            return list(zip(*map(scatter_map, obj))) if len(obj) > 0 else []
        if isinstance(obj, list):
            return list(map(list, zip(*map(scatter_map, obj)))) if len(obj
                ) > 0 else []
        if isinstance(obj, dict):
            return list(map(type(obj), zip(*map(scatter_map, obj.items())))
                ) if len(obj) > 0 else []
        return [obj for _ in device_ids]
    try:
        scattered_inputs = scatter_map(inputs)
        scattered_kwargs = scatter_map(kwargs)
        if len(scattered_inputs) < len(scattered_kwargs):
            scattered_inputs.extend([() for _ in range(len(scattered_kwargs
                ) - len(scattered_inputs))])
        elif len(scattered_kwargs) < len(scattered_inputs):
            scattered_kwargs.extend([{} for _ in range(len(scattered_inputs
                ) - len(scattered_kwargs))])
    finally:
        scatter_map = None
    return scattered_inputs, scattered_kwargs
