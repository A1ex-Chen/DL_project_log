def remove_child(self):
    """
        Closes the child progress bar.
        """
    self.child_bar = None
    self.display()
