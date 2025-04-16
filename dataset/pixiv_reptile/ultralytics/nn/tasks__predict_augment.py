def _predict_augment(self, x):
    """Perform augmentations on input image x and return augmented inference and train outputs."""
    if getattr(self, 'end2end', False):
        LOGGER.warning(
            "WARNING ⚠️ End2End model does not support 'augment=True' prediction. Reverting to single-scale prediction."
            )
        return self._predict_once(x)
    img_size = x.shape[-2:]
    s = [1, 0.83, 0.67]
    f = [None, 3, None]
    y = []
    for si, fi in zip(s, f):
        xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
        yi = super().predict(xi)[0]
        yi = self._descale_pred(yi, fi, si, img_size)
        y.append(yi)
    y = self._clip_augmented(y)
    return torch.cat(y, -1), None
