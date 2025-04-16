@property
@lru_cache(maxsize=2)
def xyxyxyxyn(self):
    """Converts rotated bounding boxes to normalized xyxyxyxy format of shape (N, 4, 2)."""
    xyxyxyxyn = self.xyxyxyxy.clone() if isinstance(self.xyxyxyxy, torch.Tensor
        ) else np.copy(self.xyxyxyxy)
    xyxyxyxyn[..., 0] /= self.orig_shape[1]
    xyxyxyxyn[..., 1] /= self.orig_shape[0]
    return xyxyxyxyn
