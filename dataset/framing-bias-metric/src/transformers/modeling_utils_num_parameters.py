def num_parameters(self, only_trainable: bool=False, exclude_embeddings:
    bool=False) ->int:
    """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            :obj:`int`: The number of parameters.
        """

    def parameter_filter(x):
        return (x.requires_grad or not only_trainable) and not (isinstance(
            x, torch.nn.Embedding) and exclude_embeddings)
    params = filter(parameter_filter, self.parameters()
        ) if only_trainable else self.parameters()
    return sum(p.numel() for p in params)
