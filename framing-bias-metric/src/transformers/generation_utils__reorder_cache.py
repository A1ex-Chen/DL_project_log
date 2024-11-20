@staticmethod
def _reorder_cache(past: Tuple[torch.Tensor], beam_idx: torch.Tensor) ->Tuple[
    torch.Tensor]:
    """
        This function is used to re-order the :obj:`past_key_values` or :obj:`mems` cache if
        :meth:`~transformers.PretrainedModel.beam_search` or :meth:`~transformers.PretrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` or :obj:`mems` with the correct beam_idx at every
        generation step.

        For custom re-ordering of :obj:`past_key_values` or :obj:`mems`, the function should be implemented in
        subclasses of :class:`~transformers.PreTrainedModel`.
        """
    return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)
