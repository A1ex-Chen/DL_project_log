def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(
            f'{model.__class__.__name__} has {total_params * 1e-06:.2f} M params.'
            )
    return total_params
