@staticmethod
def make_positions(tensor, padding_idx: int):
    """
        Replace non-padding symbols with their position numbers.

        Position numbers begin at padding_idx+1. Padding symbols are ignored.
        """
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long(
        ) + padding_idx
