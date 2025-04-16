def list(self):
    """
        List all registered metadata.

        Returns:
            list[str]: keys (names of datasets) of all registered metadata
        """
    return list(self.keys())
