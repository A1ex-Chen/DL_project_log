def _forward(x):
    """Performs a forward pass through the model, handling different Detect subclass types accordingly."""
    if self.end2end:
        return self.forward(x)['one2many']
    return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)
        ) else self.forward(x)
