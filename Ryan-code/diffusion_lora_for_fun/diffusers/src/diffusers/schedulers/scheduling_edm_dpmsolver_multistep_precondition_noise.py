def precondition_noise(self, sigma):
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor([sigma])
    c_noise = 0.25 * torch.log(sigma)
    return c_noise
