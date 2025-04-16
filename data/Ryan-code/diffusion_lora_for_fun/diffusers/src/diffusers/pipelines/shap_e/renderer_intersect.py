def intersect(self, origin: torch.Tensor, direction: torch.Tensor, t0_lower:
    Optional[torch.Tensor]=None, epsilon=1e-06):
    """
        Args:
            origin: [batch_size, *shape, 3]
            direction: [batch_size, *shape, 3]
            t0_lower: Optional [batch_size, *shape, 1] lower bound of t0 when intersecting this volume.
            params: Optional meta parameters in case Volume is parametric
            epsilon: to stabilize calculations

        Return:
            A tuple of (t0, t1, intersected) where each has a shape [batch_size, *shape, 1]. If a ray intersects with
            the volume, `o + td` is in the volume for all t in [t0, t1]. If the volume is bounded, t1 is guaranteed to
            be on the boundary of the volume.
        """
    batch_size, *shape, _ = origin.shape
    ones = [1] * len(shape)
    bbox = self.bbox.view(1, *ones, 2, 3).to(origin.device)

    def _safe_divide(a, b, epsilon=1e-06):
        return a / torch.where(b < 0, b - epsilon, b + epsilon)
    ts = _safe_divide(bbox - origin[..., None, :], direction[..., None, :],
        epsilon=epsilon)
    t0 = ts.min(dim=-2).values.max(dim=-1, keepdim=True).values.clamp(self.
        min_dist)
    t1 = ts.max(dim=-2).values.min(dim=-1, keepdim=True).values
    assert t0.shape == t1.shape == (batch_size, *shape, 1)
    if t0_lower is not None:
        assert t0.shape == t0_lower.shape
        t0 = torch.maximum(t0, t0_lower)
    intersected = t0 + self.min_t_range < t1
    t0 = torch.where(intersected, t0, torch.zeros_like(t0))
    t1 = torch.where(intersected, t1, torch.ones_like(t1))
    return VolumeRange(t0=t0, t1=t1, intersected=intersected)
