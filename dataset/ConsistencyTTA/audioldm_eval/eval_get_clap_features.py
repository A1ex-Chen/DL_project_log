def get_clap_features(pairedtextloader, clap_model):
    gt_features, gen_features, text_features, gen_mel_features = [], [], [], []
    for captions, gt_waves, gen_waves, gen_mels in tqdm(pairedtextloader):
        gt_waves = gt_waves.squeeze(1).float().to(device)
        gen_waves = gen_waves.squeeze(1).float().to(device)
        with torch.no_grad():
            seed_all(0)
            gt_features += [clap_model.get_audio_embedding_from_data(x=
                gt_waves, use_tensor=True)]
            seed_all(0)
            gen_features += [clap_model.get_audio_embedding_from_data(x=
                gen_waves, use_tensor=True)]
            seed_all(0)
            text_features += [clap_model.get_text_embedding(captions,
                use_tensor=True)]
    gt_features = torch.cat(gt_features, dim=0)
    gen_features = torch.cat(gen_features, dim=0)
    text_features = torch.cat(text_features, dim=0)
    return gt_features, gen_features, text_features
