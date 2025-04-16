def get_stats(self):
    """Returns metrics statistics and results dictionary."""
    stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}
    self.nt_per_class = np.bincount(stats['target_cls'].astype(int),
        minlength=self.nc)
    self.nt_per_image = np.bincount(stats['target_img'].astype(int),
        minlength=self.nc)
    stats.pop('target_img', None)
    if len(stats) and stats['tp'].any():
        self.metrics.process(**stats)
    return self.metrics.results_dict
