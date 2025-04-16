def put_histogram(self, hist_name, hist_tensor, bins=1000):
    """
        Create a histogram from a tensor.

        Args:
            hist_name (str): The name of the histogram to put into tensorboard.
            hist_tensor (torch.Tensor): A Tensor of arbitrary shape to be converted
                into a histogram.
            bins (int): Number of histogram bins.
        """
    ht_min, ht_max = hist_tensor.min().item(), hist_tensor.max().item()
    hist_counts = torch.histc(hist_tensor, bins=bins)
    hist_edges = torch.linspace(start=ht_min, end=ht_max, steps=bins + 1,
        dtype=torch.float32)
    hist_params = dict(tag=hist_name, min=ht_min, max=ht_max, num=len(
        hist_tensor), sum=float(hist_tensor.sum()), sum_squares=float(torch
        .sum(hist_tensor ** 2)), bucket_limits=hist_edges[1:].tolist(),
        bucket_counts=hist_counts.tolist(), global_step=self._iter)
    self._histograms.append(hist_params)
