def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module
    ]], dtype=torch.float32):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            if param.requires_grad:
                param.data = param.to(dtype)
