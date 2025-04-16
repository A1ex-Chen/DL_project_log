@torch.no_grad()
def __call__(self, input_tokens: List[List[int]], generator: Optional[torch
    .Generator]=None, num_inference_steps: int=100, return_dict: bool=True,
    output_type: str='numpy', callback: Optional[Callable[[int, int, torch.
    FloatTensor], None]]=None, callback_steps: int=1) ->Union[
    AudioPipelineOutput, Tuple]:
    if callback_steps is None or callback_steps is not None and (not
        isinstance(callback_steps, int) or callback_steps <= 0):
        raise ValueError(
            f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.'
            )
    pred_mel = np.zeros([1, TARGET_FEATURE_LENGTH, self.n_dims], dtype=np.
        float32)
    full_pred_mel = np.zeros([1, 0, self.n_dims], np.float32)
    ones = torch.ones((1, TARGET_FEATURE_LENGTH), dtype=bool, device=self.
        device)
    for i, encoder_input_tokens in enumerate(input_tokens):
        if i == 0:
            encoder_continuous_inputs = torch.from_numpy(pred_mel[:1].copy()
                ).to(device=self.device, dtype=self.decoder.dtype)
            encoder_continuous_mask = torch.zeros((1, TARGET_FEATURE_LENGTH
                ), dtype=bool, device=self.device)
        else:
            encoder_continuous_mask = ones
        encoder_continuous_inputs = self.scale_features(
            encoder_continuous_inputs, output_range=[-1.0, 1.0], clip=True)
        encodings_and_masks = self.encode(input_tokens=torch.IntTensor([
            encoder_input_tokens]).to(device=self.device),
            continuous_inputs=encoder_continuous_inputs, continuous_mask=
            encoder_continuous_mask)
        x = randn_tensor(shape=encoder_continuous_inputs.shape, generator=
            generator, device=self.device, dtype=self.decoder.dtype)
        self.scheduler.set_timesteps(num_inference_steps)
        for j, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            output = self.decode(encodings_and_masks=encodings_and_masks,
                input_tokens=x, noise_time=t / self.scheduler.config.
                num_train_timesteps)
            x = self.scheduler.step(output, t, x, generator=generator
                ).prev_sample
        mel = self.scale_to_features(x, input_range=[-1.0, 1.0])
        encoder_continuous_inputs = mel[:1]
        pred_mel = mel.cpu().float().numpy()
        full_pred_mel = np.concatenate([full_pred_mel, pred_mel[:1]], axis=1)
        if callback is not None and i % callback_steps == 0:
            callback(i, full_pred_mel)
        logger.info('Generated segment', i)
    if output_type == 'numpy' and not is_onnx_available():
        raise ValueError(
            "Cannot return output in 'np' format if ONNX is not available. Make sure to have ONNX installed or set 'output_type' to 'mel'."
            )
    elif output_type == 'numpy' and self.melgan is None:
        raise ValueError(
            "Cannot return output in 'np' format if melgan component is not defined. Make sure to define `self.melgan` or set 'output_type' to 'mel'."
            )
    if output_type == 'numpy':
        output = self.melgan(input_features=full_pred_mel.astype(np.float32))
    else:
        output = full_pred_mel
    if not return_dict:
        return output,
    return AudioPipelineOutput(audios=output)
