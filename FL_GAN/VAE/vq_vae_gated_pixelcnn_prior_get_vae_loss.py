def get_vae_loss(x, output):
    N, C, H, W = x.shape
    x_reconstructed, z_e, z_q = output
    reconstruction_loss = l2_dist(x, x_reconstructed).sum() / (N * H * W * C)
    vq_loss = l2_dist(z_e.detach(), z_q).sum() / (N * H * W * C)
    commitment_loss = l2_dist(z_e, z_q.detach()).sum() / (N * H * W * C)
    return reconstruction_loss + vq_loss + commitment_loss
