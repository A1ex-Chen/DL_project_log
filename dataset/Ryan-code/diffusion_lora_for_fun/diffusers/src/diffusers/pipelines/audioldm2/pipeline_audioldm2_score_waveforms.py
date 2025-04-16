def score_waveforms(self, text, audio, num_waveforms_per_prompt, device, dtype
    ):
    if not is_librosa_available():
        logger.info(
            'Automatic scoring of the generated audio waveforms against the input prompt text requires the `librosa` package to resample the generated waveforms. Returning the audios in the order they were generated. To enable automatic scoring, install `librosa` with: `pip install librosa`.'
            )
        return audio
    inputs = self.tokenizer(text, return_tensors='pt', padding=True)
    resampled_audio = librosa.resample(audio.numpy(), orig_sr=self.vocoder.
        config.sampling_rate, target_sr=self.feature_extractor.sampling_rate)
    inputs['input_features'] = self.feature_extractor(list(resampled_audio),
        return_tensors='pt', sampling_rate=self.feature_extractor.sampling_rate
        ).input_features.type(dtype)
    inputs = inputs.to(device)
    logits_per_text = self.text_encoder(**inputs).logits_per_text
    indices = torch.argsort(logits_per_text, dim=1, descending=True)[:, :
        num_waveforms_per_prompt]
    audio = torch.index_select(audio, 0, indices.reshape(-1).cpu())
    return audio
