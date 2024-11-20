def sample(noise=True, no_samples=100):
    """
        Sample z ~ p(z) and x ~ p(x|z) if noise, x = mu(z) otherwise
        """
    sample_shape = no_samples, latent_dim
    z = prior_p_z.sample(sample_shape).cuda()
    x_mu = vae.decoder(z)
    x_sigma = torch.ones_like(x_mu)
    x = vae.reparameterize(x_mu, x_sigma, noise)
    samples = dequantize(x, reverse=True).detach().cpu().numpy()
    return np.transpose(samples, [0, 2, 3, 1])
