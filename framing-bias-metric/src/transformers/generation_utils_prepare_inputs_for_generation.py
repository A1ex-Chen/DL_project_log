def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs
    ) ->Dict[str, Any]:
    """
        Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to prepare inputs in the
        generate method.
        """
    return {'input_ids': input_ids}
