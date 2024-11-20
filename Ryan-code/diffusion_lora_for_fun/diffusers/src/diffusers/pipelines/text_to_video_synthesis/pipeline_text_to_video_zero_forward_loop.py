def forward_loop(self, x_t0, t0, t1, generator):
    """
        Perform DDPM forward process from time t0 to t1. This is the same as adding noise with corresponding variance.

        Args:
            x_t0:
                Latent code at time t0.
            t0:
                Timestep at t0.
            t1:
                Timestamp at t1.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.

        Returns:
            x_t1:
                Forward process applied to x_t0 from time t0 to t1.
        """
    eps = randn_tensor(x_t0.size(), generator=generator, dtype=x_t0.dtype,
        device=x_t0.device)
    alpha_vec = torch.prod(self.scheduler.alphas[t0:t1])
    x_t1 = torch.sqrt(alpha_vec) * x_t0 + torch.sqrt(1 - alpha_vec) * eps
    return x_t1
