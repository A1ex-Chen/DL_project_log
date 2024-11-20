def forward(self, input: Tensor, target: Tensor, gt_wav: Tensor, captions:
    Tensor, use_ema: bool=False) ->Tensor:
    """Calculate forward propagation.
        Args:
            input (Tensor):  Predicted latent representation.
            target (Tensor): Predicted latent representation.
            groundtruth (Tensor): Ground truth audio waveform.
        Returns:
            Tensor: Weight CLAP and MSE loss.
        """
    raw_mse_loss = F.mse_loss(input.float(), target.float(), reduction='none')
    instance_mse_loss = raw_mse_loss.mean(dim=list(range(1, len(
        raw_mse_loss.shape))))
    mse_loss = reduce(instance_mse_loss, self.reduction)
    input_mel = self.vae.decode_first_stage(input.float(), allow_grad=True,
        use_ema=use_ema)
    input_wav = self.vae.decode_to_waveform(input_mel.float(), allow_grad=True)
    input_wav = input_wav[:, :int(self.sr * 10)]
    input_wav, gt_wav = tuple(resample(wav[:, :int(self.sr * 10)],
        orig_freq=self.sr, new_freq=48000, lowpass_filter_width=64, rolloff
        =0.9475937167399596, resampling_method='sinc_interp_kaiser', beta=
        14.769656459379492) for wav in (input_wav, gt_wav))
    input_feat = self.clap.get_audio_embedding_from_data(input_wav,
        use_tensor=True)
    gt_wav_feat = self.clap.get_audio_embedding_from_data(gt_wav,
        use_tensor=True)
    caption_feat = self.clap.get_text_embedding(captions, use_tensor=True)
    gen_text_similarity = F.cosine_similarity(input_feat, caption_feat, dim=1)
    gen_gt_similarity = F.cosine_similarity(input_feat, gt_wav_feat, dim=1)
    instance_loss = self.mse_weight * mse_loss + self.clap_weight * (2 -
        gen_text_similarity - gen_gt_similarity)
    return reduce(instance_loss, self.reduction)
