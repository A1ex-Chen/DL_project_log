def get_audio_embedding_from_data(self, x, use_tensor=False):
    """get audio embeddings from the audio data

        Parameters
        ----------
        x: np.darray | torch.Tensor (N,T): 
            audio data, must be mono audio tracks.
        use_tensor: boolean:
            if True, x should be the tensor input and the output will be the tesnor, preserving the gradient (default: False).      
            Note that if 'use tensor' is set to True, it will not do the quantize of the audio waveform (otherwise the gradient will not be preserved).
        Returns
        ----------
        audio embed: numpy.darray | torch.Tensor (N,D):
            audio embeddings that extracted from audio files
        """
    self.model.eval()
    audio_input = []
    for audio_waveform in x:
        if not use_tensor:
            audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
            audio_waveform = torch.from_numpy(audio_waveform).float()
        temp_dict = {}
        temp_dict = get_audio_features(temp_dict, audio_waveform, max_len=
            480000, data_truncating='fusion' if self.enable_fusion else
            'rand_trunc', data_filling='repeatpad', audio_cfg=self.
            model_cfg['audio_cfg'], require_grad=audio_waveform.requires_grad)
        audio_input.append(temp_dict)
    audio_embed = self.model.get_audio_embedding(audio_input)
    if not use_tensor:
        audio_embed = audio_embed.detach().cpu().numpy()
    return audio_embed
