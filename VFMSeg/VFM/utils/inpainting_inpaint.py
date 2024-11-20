@torch.no_grad()
def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps,
    num_samples=1, w=512, h=512):
    device = torch.device('cuda') if torch.cuda.is_available(
        ) else torch.device('cpu')
    model = sampler.model
    print(
        'Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...'
        )
    wm = 'SDV2'
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch
        .float32)
    with torch.no_grad(), torch.autocast('cuda'):
        batch = make_batch_sd(image, mask, txt=prompt, device=device,
            num_samples=num_samples)
        c = model.cond_stage_model.encode(batch['txt'])
        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck != model.masked_image_key:
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.get_first_stage_encoding(model.
                    encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        cond = {'c_concat': [c_cat], 'c_crossattn': [c]}
        uc_cross = model.get_unconditional_conditioning(num_samples, '')
        uc_full = {'c_concat': [c_cat], 'c_crossattn': [uc_cross]}
        shape = [model.channels, h // 8, w // 8]
        samples_cfg, intermediates = sampler.sample(ddim_steps, num_samples,
            shape, cond, verbose=False, eta=1.0,
            unconditional_guidance_scale=scale, unconditional_conditioning=
            uc_full, x_T=start_code)
        x_samples_ddim = model.decode_first_stage(samples_cfg)
        result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder
        ) for img in result]
