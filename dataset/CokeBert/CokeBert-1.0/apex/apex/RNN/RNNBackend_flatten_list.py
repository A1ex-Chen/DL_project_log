def flatten_list(tens_list):
    """
    flatten_list
    """
    if not is_iterable(tens_list):
        return tens_list
    return torch.cat(tens_list, dim=0).view(len(tens_list), *tens_list[0].
        size())
