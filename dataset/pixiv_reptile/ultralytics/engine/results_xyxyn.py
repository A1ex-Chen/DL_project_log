@property
@lru_cache(maxsize=2)
def xyxyn(self):
    """Normalize box coordinates to [x1, y1, x2, y2] relative to the original image size."""
    xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor
        ) else np.copy(self.xyxy)
    xyxy[..., [0, 2]] /= self.orig_shape[1]
    xyxy[..., [1, 3]] /= self.orig_shape[0]
    return xyxy
