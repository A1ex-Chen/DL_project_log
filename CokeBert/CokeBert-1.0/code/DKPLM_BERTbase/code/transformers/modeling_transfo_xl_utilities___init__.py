def __init__(self, range_max, n_sample):
    """
        Reference : https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py
            `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`

        expected count can be approximated by 1 - (1 - p)^n
        and we use a numerically stable version -expm1(num_tries * log1p(-p))

        Our implementation fixes num_tries at 2 * n_sample, and the actual #samples will vary from run to run
        """
    with torch.no_grad():
        self.range_max = range_max
        log_indices = torch.arange(1.0, range_max + 2.0, 1.0).log_()
        self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]
        self.log_q = (-(-self.dist.double().log1p_() * 2 * n_sample).expm1_()
            ).log_().float()
    self.n_sample = n_sample
