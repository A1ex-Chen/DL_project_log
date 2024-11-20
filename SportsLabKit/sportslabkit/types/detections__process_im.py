def _process_im(self, im: (str | Path | Image.Image | np.ndarray)
    ) ->np.ndarray:
    return read_image(im)
