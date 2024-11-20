def adjust_logits_during_generation(self, logits, **kwargs):
    """
        Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to adjust the logits in
        the generate method.
        """
    return logits
