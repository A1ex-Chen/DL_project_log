def sample(self, labels):
    """
            labels: [b1, b2]
        Return
            true_log_probs: [b1, b2]
            samp_log_probs: [n_sample]
            neg_samples: [n_sample]
        """
    n_sample = self.n_sample
    n_tries = 2 * n_sample
    with torch.no_grad():
        neg_samples = torch.multinomial(self.dist, n_tries, replacement=True
            ).unique()
        device = labels.device
        neg_samples = neg_samples.to(device)
        true_log_probs = self.log_q[labels].to(device)
        samp_log_probs = self.log_q[neg_samples].to(device)
        return true_log_probs, samp_log_probs, neg_samples
