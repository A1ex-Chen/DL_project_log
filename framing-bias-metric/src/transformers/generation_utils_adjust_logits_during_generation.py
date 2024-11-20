def adjust_logits_during_generation(self, logits: torch.FloatTensor, **kwargs
    ) ->torch.FloatTensor:
    """
        Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to adjust the logits in
        the generate method.
        """
    return logits
