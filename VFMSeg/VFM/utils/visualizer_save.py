def save(self, filepath):
    """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
    self.fig.savefig(filepath)
