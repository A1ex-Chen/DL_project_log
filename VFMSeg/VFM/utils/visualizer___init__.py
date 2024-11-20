def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=
    ColorMode.IMAGE):
    """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            metadata (Metadata): dataset metadata (e.g. class names and colors)
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
        """
    self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
    if metadata is None:
        metadata = MetadataCatalog.get('__nonexist__')
    self.metadata = metadata
    self.output = VisImage(self.img, scale=scale)
    self.cpu_device = torch.device('cpu')
    self._default_font_size = max(np.sqrt(self.output.height * self.output.
        width) // 90, 10 // scale)
    self._default_font_size = 18
    self._instance_mode = instance_mode
    self.keypoint_threshold = _KEYPOINT_THRESHOLD
