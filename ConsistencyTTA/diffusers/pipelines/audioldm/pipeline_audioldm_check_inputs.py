def check_inputs(self, prompt, audio_length_in_s, vocoder_upsample_factor,
    callback_steps, negative_prompt=None, prompt_embeds=None,
    negative_prompt_embeds=None):
    min_audio_length_in_s = vocoder_upsample_factor * self.vae_scale_factor
    if audio_length_in_s < min_audio_length_in_s:
        raise ValueError(
            f'`audio_length_in_s` has to be a positive value greater than or equal to {min_audio_length_in_s}, but is {audio_length_in_s}.'
            )
    if self.vocoder.config.model_in_dim % self.vae_scale_factor != 0:
        raise ValueError(
            f"The number of frequency bins in the vocoder's log-mel spectrogram has to be divisible by the VAE scale factor, but got {self.vocoder.config.model_in_dim} bins and a scale factor of {self.vae_scale_factor}."
            )
    if callback_steps is None or callback_steps is not None and (not
        isinstance(callback_steps, int) or callback_steps <= 0):
        raise ValueError(
            f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.'
            )
    if prompt is not None and prompt_embeds is not None:
        raise ValueError(
            f'Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two.'
            )
    elif prompt is None and prompt_embeds is None:
        raise ValueError(
            'Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.'
            )
    elif prompt is not None and (not isinstance(prompt, str) and not
        isinstance(prompt, list)):
        raise ValueError(
            f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
    if negative_prompt is not None and negative_prompt_embeds is not None:
        raise ValueError(
            f'Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to only forward one of the two.'
            )
    if prompt_embeds is not None and negative_prompt_embeds is not None:
        if prompt_embeds.shape != negative_prompt_embeds.shape:
            raise ValueError(
                f'`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds` {negative_prompt_embeds.shape}.'
                )
