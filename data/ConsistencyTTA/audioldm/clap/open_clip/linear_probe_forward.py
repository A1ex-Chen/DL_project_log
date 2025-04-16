def forward(self, x, mix_lambda=None, device=None):
    """
        Args:
            x: waveform, torch.tensor [batch, t_samples] / batch of mel_spec and longer list
            mix_lambda: torch.tensor [batch], the mixup lambda
        Returns:
            class_prob: torch.tensor [batch, class_num]

        """
    if self.freeze:
        self.clap_model.eval()
    x = self.clap_model.audio_projection(self.clap_model.audio_branch(x,
        mixup_lambda=mix_lambda, device=device)['embedding'])
    out = self.lp_layer(x)
    if self.act is not None:
        out = self.act(out)
    return out
