def _get_lines(self, image):
    """Detect lines in the image using Fast Line Detector."""
    lines = self.fld.detect(image)
    return lines
