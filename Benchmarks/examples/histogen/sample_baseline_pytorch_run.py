def run(params):
    get_data(params)
    args = candle.ArgumentStruct(**params)
    if args.use_gpus:
        device_ids = []
        ndevices = torch.cuda.device_count()
        if ndevices > 1:
            for i in range(ndevices):
                device_i = torch.device('cuda:' + str(i))
                device_ids.append(device_i)
            device = device_ids[0]
        elif ndevices == 1:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    model_vqvae = load_model('vqvae', args.vqvae, device)
    model_top = load_model('pixelsnail_top', args.top, device)
    model_bottom = load_model('pixelsnail_bottom', args.bottom, device)
    top_sample = sample_model(model_top, device, args.batch_size, [32, 32],
        args.temp)
    bottom_sample = sample_model(model_bottom, device, args.batch_size, [64,
        64], args.temp, condition=top_sample)
    decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
    decoded_sample = decoded_sample.clamp(-1, 1)
    save_image(decoded_sample, args.filename, normalize=True, range=(-1, 1))
