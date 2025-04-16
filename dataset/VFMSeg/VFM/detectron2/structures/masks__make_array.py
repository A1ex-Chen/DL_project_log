def _make_array(t: Union[torch.Tensor, np.ndarray]) ->np.ndarray:
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()
    return np.asarray(t).astype('float64')
