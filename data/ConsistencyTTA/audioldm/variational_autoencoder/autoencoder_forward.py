def forward(self, input, sample_posterior=True):
    posterior = self.encode(input)
    z = posterior.sample() if sample_posterior else posterior.mode()
    if self.flag_first_run:
        print('Latent size: ', z.size())
        self.flag_first_run = False
    return self.decode(z), posterior
