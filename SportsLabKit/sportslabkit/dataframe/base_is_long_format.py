def is_long_format(self):
    """Check if the dataframe is in long format.

        Returns:
            bool: True if the dataframe is in long format, False otherwise.
        """
    return self.index.nlevels == 3
