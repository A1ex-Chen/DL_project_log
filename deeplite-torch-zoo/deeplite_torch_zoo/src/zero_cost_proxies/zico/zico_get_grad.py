def get_grad(model: torch.nn.Module, grad_dict: dict, step_iter=0):
    if step_iter == 0:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                grad_dict[name] = [module.weight.grad.data.cpu().reshape(-1
                    ).numpy()]
    else:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                grad_dict[name].append(module.weight.grad.data.cpu().
                    reshape(-1).numpy())
    return grad_dict
