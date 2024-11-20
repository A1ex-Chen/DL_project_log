def load(self, filename):
    """Loads a module dictionary from local file or url.

        Args:
            filename (str): name of saved module dictionary
        """
    if is_url(filename):
        return self.load_url(filename)
    else:
        return self.load_file(filename)
