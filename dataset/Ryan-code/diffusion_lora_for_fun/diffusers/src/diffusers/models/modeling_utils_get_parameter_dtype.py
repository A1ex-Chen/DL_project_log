def get_parameter_dtype(parameter: torch.nn.Module) ->torch.dtype:
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype
        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype
    except StopIteration:

        def find_tensor_attributes(module: torch.nn.Module) ->List[Tuple[
            str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.
                is_tensor(v)]
            return tuples
        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype
