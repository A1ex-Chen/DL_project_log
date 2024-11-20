def nested_numpify(tensors):
    """Numpify `tensors` (even if it's a nested list/tuple of tensors)."""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    return tensors.cpu().numpy()
