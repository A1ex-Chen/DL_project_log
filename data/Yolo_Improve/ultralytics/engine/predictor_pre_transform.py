def pre_transform(self, im):
    """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
    same_shapes = len({x.shape for x in im}) == 1
    letterbox = LetterBox(self.imgsz, auto=same_shapes and self.model.pt,
        stride=self.model.stride)
    return [letterbox(image=x) for x in im]
