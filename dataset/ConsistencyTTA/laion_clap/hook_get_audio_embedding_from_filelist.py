def get_audio_embedding_from_filelist(self, x, use_tensor=False):
    """get audio embeddings from the audio file list

        Parameters
        ----------
        x: List[str] (N,): 
            an audio file list to extract features, audio files can have different lengths (as we have the feature fusion machanism)
        use_tensor: boolean:
            if True, it will return the torch tensor, preserving the gradient (default: False).
        Returns
        ----------
        audio_embed : numpy.darray | torch.Tensor (N,D):
            audio embeddings that extracted from audio files
        """
    self.model.eval()
    audio_input = []
    for f in x:
        audio_waveform, _ = librosa.load(f, sr=48000)
        audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
        audio_waveform = torch.from_numpy(audio_waveform).float()
        temp_dict = {}
        temp_dict = get_audio_features(temp_dict, audio_waveform, 480000,
            data_truncating='fusion' if self.enable_fusion else
            'rand_trunc', data_filling='repeatpad', audio_cfg=self.
            model_cfg['audio_cfg'], require_grad=audio_waveform.requires_grad)
        audio_input.append(temp_dict)
    audio_embed = self.model.get_audio_embedding(audio_input)
    if not use_tensor:
        audio_embed = audio_embed.detach().cpu().numpy()
    return audio_embed
