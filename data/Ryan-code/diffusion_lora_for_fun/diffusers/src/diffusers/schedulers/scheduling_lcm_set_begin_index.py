def set_begin_index(self, begin_index: int=0):
    """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
    self._begin_index = begin_index
