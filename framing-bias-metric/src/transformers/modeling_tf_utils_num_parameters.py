def num_parameters(self, only_trainable: bool=False) ->int:
    """
        Get the number of (optionally, trainable) parameters in the model.

        Args:
            only_trainable (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return only the number of trainable parameters

        Returns:
            :obj:`int`: The number of parameters.
        """
    if only_trainable:
        return int(sum(np.prod(w.shape.as_list()) for w in self.
            trainable_variables))
    else:
        return self.count_params()
