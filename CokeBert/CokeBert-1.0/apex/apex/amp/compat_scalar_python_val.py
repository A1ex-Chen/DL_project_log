def scalar_python_val(x):
    if hasattr(x, 'item'):
        return x.item()
    elif isinstance(x, torch.autograd.Variable):
        return x.data[0]
    else:
        return x[0]
