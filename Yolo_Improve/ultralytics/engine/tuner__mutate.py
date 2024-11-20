def _mutate(self, parent='single', n=5, mutation=0.8, sigma=0.2):
    """
        Mutates the hyperparameters based on bounds and scaling factors specified in `self.space`.

        Args:
            parent (str): Parent selection method: 'single' or 'weighted'.
            n (int): Number of parents to consider.
            mutation (float): Probability of a parameter mutation in any given iteration.
            sigma (float): Standard deviation for Gaussian random number generator.

        Returns:
            (dict): A dictionary containing mutated hyperparameters.
        """
    if self.tune_csv.exists():
        x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=',', skiprows=1)
        fitness = x[:, 0]
        n = min(n, len(x))
        x = x[np.argsort(-fitness)][:n]
        w = x[:, 0] - x[:, 0].min() + 1e-06
        if parent == 'single' or len(x) == 1:
            x = x[random.choices(range(n), weights=w)[0]]
        elif parent == 'weighted':
            x = (x * w.reshape(n, 1)).sum(0) / w.sum()
        r = np.random
        r.seed(int(time.time()))
        g = np.array([(v[2] if len(v) == 3 else 1.0) for k, v in self.space
            .items()])
        ng = len(self.space)
        v = np.ones(ng)
        while all(v == 1):
            v = (g * (r.random(ng) < mutation) * r.randn(ng) * r.random() *
                sigma + 1).clip(0.3, 3.0)
        hyp = {k: float(x[i + 1] * v[i]) for i, k in enumerate(self.space.
            keys())}
    else:
        hyp = {k: getattr(self.args, k) for k in self.space.keys()}
    for k, v in self.space.items():
        hyp[k] = max(hyp[k], v[0])
        hyp[k] = min(hyp[k], v[1])
        hyp[k] = round(hyp[k], 5)
    return hyp
