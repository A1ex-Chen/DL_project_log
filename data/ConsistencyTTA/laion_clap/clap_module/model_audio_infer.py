def audio_infer(self, audio, hopsize=None, device=None):
    """Forward one audio and produce the audio embedding

        Parameters
        ----------
        audio:  (audio_length)
            the time-domain audio input, notice that it must be only one input
        hopsize: int
            the overlap hopsize as the sliding window

        Returns
        ----------
        output_dict: {
            key: [n, (embedding_shape)] if "HTS-AT"
            or
            key: [(embedding_shape)] if "PANN"
        }
            the list of key values of the audio branch

        """
    assert not self.training, 'the inference mode must be run at eval stage'
    output_dict = {}
    if self.audio_cfg.model_type == 'PANN':
        audio_input = audio.unsqueeze(dim=0)
        output_dict[key] = self.encode_audio(audio_input, device=device)[key
            ].squeeze(dim=0)
    elif self.audio_cfg.model_type == 'HTSAT':
        audio_len = len(audio)
        k = self.audio_cfg.clip_samples // audio_len
        if k > 1:
            audio = audio.repeat(k)
            audio_len = len(audio)
        if hopsize is None:
            hopsize = min(hopsize, audio_len)
        if audio_len > self.audio_cfg.clip_samples:
            audio_input = [audio[pos:pos + self.audio_cfg.clip_samples].
                clone() for pos in range(0, audio_len - self.audio_cfg.
                clip_samples, hopsize)]
            audio_input.append(audio[-self.audio_cfg.clip_samples:].clone())
            audio_input = torch.stack(audio_input)
            output_dict[key] = self.encode_audio(audio_input, device=device)[
                key]
        else:
            audio_input = audio.unsqueeze(dim=0)
            output_dict[key] = self.encode_audio(audio_input, device=device)[
                key].squeeze(dim=0)
    return output_dict
