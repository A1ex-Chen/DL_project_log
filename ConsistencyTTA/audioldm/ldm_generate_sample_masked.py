@torch.no_grad()
def generate_sample_masked(self, batchs, ddim_steps=200, ddim_eta=1.0, x_T=
    None, n_candidate_gen_per_text=1, unconditional_guidance_scale=1.0,
    unconditional_conditioning=None, name='waveform', use_plms=False,
    time_mask_ratio_start_and_end=(0.25, 0.75),
    freq_mask_ratio_start_and_end=(0.75, 1.0), save=False, **kwargs):
    assert x_T is None
    try:
        batchs = iter(batchs)
    except TypeError:
        raise ValueError(
            'The first input argument should be an iterable object')
    if use_plms:
        assert ddim_steps is not None
    use_ddim = ddim_steps is not None
    with self.ema_scope('Generate'):
        for batch in batchs:
            z, c = self.get_input(batch, self.first_stage_key, cond_key=
                self.cond_stage_key, return_first_stage_outputs=False,
                force_c_encode=True, return_original_cond=False, bs=None)
            text = super().get_input(batch, 'text')
            batch_size = z.shape[0] * n_candidate_gen_per_text
            _, h, w = z.shape[0], z.shape[2], z.shape[3]
            mask = torch.ones(batch_size, h, w).to(self.device)
            mask[:, int(h * time_mask_ratio_start_and_end[0]):int(h *
                time_mask_ratio_start_and_end[1]), :] = 0
            mask[:, :, int(w * freq_mask_ratio_start_and_end[0]):int(w *
                freq_mask_ratio_start_and_end[1])] = 0
            mask = mask[:, None, ...]
            c = torch.cat([c] * n_candidate_gen_per_text, dim=0)
            text = text * n_candidate_gen_per_text
            if unconditional_guidance_scale != 1.0:
                unconditional_conditioning = (self.cond_stage_model.
                    get_unconditional_condition(batch_size))
            samples, _ = self.sample_log(cond=c, batch_size=batch_size, x_T
                =x_T, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                use_plms=use_plms, mask=mask, x0=torch.cat([z] *
                n_candidate_gen_per_text))
            mel = self.decode_first_stage(samples)
            waveform = self.mel_spectrogram_to_waveform(mel)
            if waveform.shape[0] > 1:
                similarity = self.cond_stage_model.cos_similarity(torch.
                    FloatTensor(waveform).squeeze(1), text)
                best_index = []
                for i in range(z.shape[0]):
                    candidates = similarity[i::z.shape[0]]
                    max_index = torch.argmax(candidates).item()
                    best_index.append(i + max_index * z.shape[0])
                waveform = waveform[best_index]
    return waveform
