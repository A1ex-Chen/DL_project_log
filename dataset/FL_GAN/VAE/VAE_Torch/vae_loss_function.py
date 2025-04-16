def loss_function(batch, output):

    def repeat(tensor, K=50):
        shape = (K,) + tuple(tensor.shape)
        return torch.cat(K * [tensor]).reshape(shape)
    K = 50
    x_mu, x_sigma, z, z_mu, z_sigma = output
    x_mu, x_sigma, z, z_mu, z_sigma = repeat(x_mu), repeat(x_sigma), repeat(z
        ), repeat(z_mu), repeat(z_sigma)
    k_batch = repeat(batch)
    log_p_z = prior_p_z.log_prob(z)
    z_normal = dist.normal.Normal(z_mu, z_sigma)
    posterior_log_prob = z_normal.log_prob(z)
    x_normal = dist.normal.Normal(x_mu, x_sigma)
    x_reconstruct_log_prob = x_normal.log_prob(k_batch)
    VLB = -torch.mean(log_p_z.sum(dim=(0, 2)) + x_reconstruct_log_prob.sum(
        dim=(0, 2, 3, 4)) - posterior_log_prob.sum(dim=(0, 2))) / K
    reconstruction_loss = -torch.mean(x_reconstruct_log_prob.sum(dim=(0, 2,
        3, 4))) / K
    KL = torch.mean(posterior_log_prob.sum(dim=(0, 2)) - log_p_z.sum(dim=(0,
        2))) / K
    return VLB, reconstruction_loss, KL
