def text_to_audio(latent_diffusion, text, original_audio_file_path=None,
    seed=42, ddim_steps=200, duration=10, batchsize=1, guidance_scale=2.5,
    n_candidate_gen_per_text=3, config=None):
    seed_everything(int(seed))
    waveform = None
    if original_audio_file_path is not None:
        waveform = read_wav_file(original_audio_file_path, int(duration * 
            102.4) * 160)
    batch = make_batch_for_text_to_audio(text, waveform=waveform, batchsize
        =batchsize)
    latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)
    if waveform is not None:
        print('Generate audio that has similar content as %s' %
            original_audio_file_path)
        latent_diffusion = set_cond_audio(latent_diffusion)
    else:
        print('Generate audio using text %s' % text)
        latent_diffusion = set_cond_text(latent_diffusion)
    with torch.no_grad():
        waveform = latent_diffusion.generate_sample([batch],
            unconditional_guidance_scale=guidance_scale, ddim_steps=
            ddim_steps, n_candidate_gen_per_text=n_candidate_gen_per_text,
            duration=duration)
    return waveform
