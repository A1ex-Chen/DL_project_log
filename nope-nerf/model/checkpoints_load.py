def load(self, filename, device=None, load_model_only=False):
    """Loads a module dictionary from local file or url.

        Args:
            filename (str): name of saved module dictionary
        """
    if is_url(filename):
        return self.load_url(filename)
    else:
        return self.load_file(filename, device, load_model_only)
