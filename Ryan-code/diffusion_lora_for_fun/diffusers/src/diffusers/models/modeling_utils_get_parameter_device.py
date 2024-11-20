def get_parameter_device(parameter: torch.nn.Module) ->torch.device:
    try:
        parameters_and_buffers = itertools.chain(parameter.parameters(),
            parameter.buffers())
        return next(parameters_and_buffers).device
    except StopIteration:

        def find_tensor_attributes(module: torch.nn.Module) ->List[Tuple[
            str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.
                is_tensor(v)]
            return tuples
        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device
