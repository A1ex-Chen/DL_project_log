def add_child(self, total, prefix=None, width=300):
    """
        Add a child progress bar displayed under the table of metrics. The child progress bar is returned (so it can be
        easily updated).

        Args:
            total (:obj:`int`): The number of iterations for the child progress bar.
            prefix (:obj:`str`, `optional`): A prefix to write on the left of the progress bar.
            width (:obj:`int`, `optional`, defaults to 300): The width (in pixels) of the progress bar.
        """
    self.child_bar = NotebookProgressBar(total, prefix=prefix, parent=self,
        width=width)
    return self.child_bar
