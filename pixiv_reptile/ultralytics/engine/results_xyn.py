@property
@lru_cache(maxsize=1)
def xyn(self):
    """Returns normalized coordinates (x, y) of keypoints relative to the original image size."""
    xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self
        .xy)
    xy[..., 0] /= self.orig_shape[1]
    xy[..., 1] /= self.orig_shape[0]
    return xy
