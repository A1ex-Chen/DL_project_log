def pop_callback(self, callback):
    """
        Remove a callback from the current list of :class:`~transformer.TrainerCallback` and returns it.

        If the callback is not found, returns :obj:`None` (and no error is raised).

        Args:
           callback (:obj:`type` or :class:`~transformer.TrainerCallback`):
               A :class:`~transformer.TrainerCallback` class or an instance of a :class:`~transformer.TrainerCallback`.
               In the first case, will pop the first member of that class found in the list of callbacks.

        Returns:
            :class:`~transformer.TrainerCallback`: The callback removed, if found.
        """
    return self.callback_handler.pop_callback(callback)
