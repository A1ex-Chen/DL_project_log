def infer_z(self, inputs, c=None, data=None):
    """ Infers a latent code z.

        The inputs and latent conditioned code are passed to the latent encoder
        to obtain the predicted mean and standard deviation.

        Args:
            inputs (tensor): input tensor
            c (tensor): latent conditioned code c
        """
    if self.encoder_latent is not None:
        mean_z, logstd_z = self.encoder_latent(inputs, c, data=data)
    else:
        batch_size = inputs.size(0)
        mean_z = torch.empty(batch_size, 0).to(self.device)
        logstd_z = torch.empty(batch_size, 0).to(self.device)
    q_z = dist.Normal(mean_z, torch.exp(logstd_z))
    if self.encoder_latent_temporal is not None:
        mean_z, logstd_z = self.encoder_latent_temporal(inputs, c)
    q_z_t = dist.Normal(mean_z, torch.exp(logstd_z))
    return q_z, q_z_t
