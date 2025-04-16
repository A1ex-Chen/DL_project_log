def _convert_ndarray_to_tensor(state_dict: Dict[str, Any]) ->None:
    """
    In-place convert all numpy arrays in the state_dict to torch tensor.
    Args:
        state_dict (dict): a state-dict to be loaded to the model.
            Will be modified.
    """
    for k in list(state_dict.keys()):
        v = state_dict[k]
        if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
            raise ValueError('Unsupported type found in checkpoint! {}: {}'
                .format(k, type(v)))
        if not isinstance(v, torch.Tensor):
            state_dict[k] = torch.from_numpy(v)
