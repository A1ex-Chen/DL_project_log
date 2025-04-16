def pre_process(self, img_src, input_shape=None):
    """Preprocess an image before TRT YOLO inferencing.
        """
    input_shape = input_shape if input_shape is not None else self.input_shape
    image, ratio, pad = letterbox(img_src, input_shape, auto=False,
        return_int=self.return_int, scaleup=True)
    image = image.transpose((2, 0, 1))[::-1]
    image = torch.from_numpy(np.ascontiguousarray(image)).to(self.device
        ).float()
    image = image / 255.0
    return image, pad
