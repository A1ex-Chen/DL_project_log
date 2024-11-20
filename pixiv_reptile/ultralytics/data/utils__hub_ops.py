def _hub_ops(self, f):
    """Saves a compressed image for HUB previews."""
    compress_one_image(f, self.im_dir / Path(f).name)
