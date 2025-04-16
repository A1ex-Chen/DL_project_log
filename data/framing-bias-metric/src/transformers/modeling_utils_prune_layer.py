def prune_layer(layer: Union[torch.nn.Linear, Conv1D], index: torch.
    LongTensor, dim: Optional[int]=None) ->Union[torch.nn.Linear, Conv1D]:
    """
    Prune a Conv1D or linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (:obj:`Union[torch.nn.Linear, Conv1D]`): The layer to prune.
        index (:obj:`torch.LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`): The dimension on which to keep the indices.

    Returns:
        :obj:`torch.nn.Linear` or :class:`~transformers.modeling_utils.Conv1D`: The pruned layer as a new layer with
        :obj:`requires_grad=True`.
    """
    if isinstance(layer, nn.Linear):
        return prune_linear_layer(layer, index, dim=0 if dim is None else dim)
    elif isinstance(layer, Conv1D):
        return prune_conv1d_layer(layer, index, dim=1 if dim is None else dim)
    else:
        raise ValueError("Can't prune layer of class {}".format(layer.
            __class__))
