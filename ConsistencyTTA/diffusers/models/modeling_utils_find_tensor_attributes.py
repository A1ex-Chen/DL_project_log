def find_tensor_attributes(module: torch.nn.Module) ->List[Tuple[str, Tensor]]:
    tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
    return tuples
