def forward(self, audio, text, device=None):
    """Forward audio and text into the CLAP

        Parameters
        ----------
        audio: torch.Tensor (batch_size, audio_length)
            the time-domain audio input / the batch of mel_spec and longer list.
        text: torch.Tensor () // need to add
            the text token input
        """
    if device is None:
        if audio is not None:
            device = audio.device
        elif text is not None:
            device = text.device
    if audio is None and text is None:
        return self.logit_scale_a.exp(), self.logit_scale_t.exp()
    elif audio is None:
        return self.encode_text(text, device=device)
    elif text is None:
        return self.audio_projection(self.encode_audio(audio, device=device
            )['embedding'])
    audio_features = self.audio_projection(self.encode_audio(audio, device=
        device)['embedding'])
    audio_features = F.normalize(audio_features, dim=-1)
    text_features = self.encode_text(text, device=device)
    text_features = F.normalize(text_features, dim=-1)
    audio_features_mlp = self.audio_transform(audio_features)
    text_features_mlp = self.text_transform(text_features)
    return (audio_features, text_features, audio_features_mlp,
        text_features_mlp, self.logit_scale_a.exp(), self.logit_scale_t.exp())
