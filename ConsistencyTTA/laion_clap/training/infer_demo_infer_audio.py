def infer_audio():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    precision = 'fp32'
    amodel = 'HTSAT-tiny'
    tmodel = 'roberta'
    enable_fusion = True
    fusion_type = 'aff_2d'
    pretrained = '/home/la/kechen/Research/KE_CLAP/ckpt/fusion_best.pt'
    model, model_cfg = create_model(amodel, tmodel, pretrained, precision=
        precision, device=device, enable_fusion=enable_fusion, fusion_type=
        fusion_type)
    audio_waveform, sr = librosa.load(
        '/home/la/kechen/Research/KE_CLAP/ckpt/test_clap_short.wav', sr=48000)
    audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
    audio_waveform = torch.from_numpy(audio_waveform).float()
    audio_dict = {}
    audio_dict = get_audio_features(audio_dict, audio_waveform, 480000,
        data_truncating='fusion', data_filling='repeatpad', audio_cfg=
        model_cfg['audio_cfg'])
    model.eval()
    audio_embed = model.get_audio_embedding([audio_dict])
    audio_embed = audio_embed.detach().cpu().numpy()
    print(audio_embed)
    print(audio_embed.shape)
