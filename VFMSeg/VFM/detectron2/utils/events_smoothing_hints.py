def smoothing_hints(self):
    """
        Returns:
            dict[name -> bool]: the user-provided hint on whether the scalar
                is noisy and needs smoothing.
        """
    return self._smoothing_hints
