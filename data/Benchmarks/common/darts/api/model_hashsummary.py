def hashsummary(self):
    """Print a model summary - checksums of each layer parameters"""
    children = list(self.children())
    result = []
    for child in children:
        result.extend(hashlib.sha256(x.detach().cpu().numpy().tobytes()).
            hexdigest() for x in child.parameters())
    return result
