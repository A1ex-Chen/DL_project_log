def show(self, title=None):
    """Show the annotated image."""
    Image.fromarray(np.asarray(self.im)[..., ::-1]).show(title)
