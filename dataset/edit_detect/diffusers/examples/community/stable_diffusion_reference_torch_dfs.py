def torch_dfs(model: torch.nn.Module):
    """
    Performs a depth-first search on the given PyTorch model and returns a list of all its child modules.

    Args:
        model (torch.nn.Module): The PyTorch model to perform the depth-first search on.

    Returns:
        list: A list of all child modules of the given model.
    """
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result
