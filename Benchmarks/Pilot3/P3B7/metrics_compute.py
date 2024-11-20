def compute(self):
    """Compute the F1 score over all batches"""
    return {t: f1.compute().item() for t, f1 in self.metrics.items()}
