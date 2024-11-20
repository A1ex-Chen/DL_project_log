def __init__(self, preds: (list[dict] | list[list] | list[Detection]), im:
    (str | Path | Image.Image | np.ndarray), times: tuple[float, float,
    float]=(0, 0, 0), names: (list[str] | None)=None):
    self.im = self._process_im(im)
    self.preds = self._process_preds(preds)
    self.names = names
    self.times = times
