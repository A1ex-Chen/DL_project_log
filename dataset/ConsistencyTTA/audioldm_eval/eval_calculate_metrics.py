def calculate_metrics(self, dataset_json_path, generate_files_path,
    groundtruth_path, mel_path, same_name, target_length=1000, limit_num=None):
    seed_all(0)
    print(f'generate_files_path: {generate_files_path}')
    print(f'groundtruth_path: {groundtruth_path}')
    print('Checking file integrity...')
    generated_files = [f for f in os.listdir(generate_files_path) if f.
        endswith('.wav')]
    groundtruth_files = [f for f in os.listdir(groundtruth_path) if f.
        endswith('.wav')]
    if generated_files != groundtruth_files:
        print(
            f'In generated but not in groundtruth: {set(generated_files) - set(groundtruth_files)}'
            )
        print(
            f'In groundtruth but not in generated: {set(groundtruth_files) - set(generated_files)}'
            )
        raise ValueError(
            f"""Generated and groundtruth diretories have different files.
Generated: {generated_files};
Ground truth: {groundtruth_files}."""
            )
    print('Done checking files.')
    outputloader, resultloader = tuple([DataLoader(WaveDataset(path, self.
        sampling_rate, limit_num=limit_num, target_length=tl), batch_size=1,
        sampler=None, num_workers=4) for path, tl in zip([
        generate_files_path, groundtruth_path], [target_length, 1000])])
    pairedtextdataset = T2APairedDataset(dataset_json_path=
        dataset_json_path, generated_path=generate_files_path,
        target_length=target_length, mel_path=mel_path, sample_rate=[16000,
        48000])
    pairedtextloader = DataLoader(pairedtextdataset, batch_size=16,
        num_workers=8, shuffle=False, collate_fn=pairedtextdataset.collate_fn)
    melpaireddataset = MelPairedDataset(generate_files_path,
        groundtruth_path, self._stft, self.sampling_rate, self.fbin_mean,
        self.fbin_std, limit_num=limit_num)
    melpairedloader = DataLoader(melpaireddataset, batch_size=1, sampler=
        None, num_workers=8, shuffle=False)
    out = {}
    print('Calculating Frechet Audio Distance...')
    fad_score = self.frechet.score(generate_files_path, groundtruth_path,
        target_length=target_length)
    out.update(fad_score)
    print('Calculating LSD...')
    metric_lsd = self.calculate_lsd(melpairedloader, same_name=same_name)
    out.update(metric_lsd)
    print('Calculating CLAP score...')
    gt_feat, gen_feat, text_feat = get_clap_features(pairedtextloader, self
        .clap_model)
    gt_text_similarity = cosine_similarity(gt_feat, text_feat, dim=1)
    gen_text_similarity = cosine_similarity(gen_feat, text_feat, dim=1)
    gen_gt_similarity = cosine_similarity(gen_feat, gt_feat, dim=1)
    gt_text_similarity = torch.clamp(gt_text_similarity, min=0)
    gen_text_similarity = torch.clamp(gen_text_similarity, min=0)
    gen_gt_similarity = torch.clamp(gen_gt_similarity, min=0)
    out.update({'gt_text_clap_score': gt_text_similarity.mean().item() * 
        100.0, 'gen_text_clap_score': gen_text_similarity.mean().item() * 
        100.0, 'gen_gt_clap_score': gen_gt_similarity.mean().item() * 100.0})
    print('Calculating PSNR...')
    metric_psnr_ssim = self.calculate_psnr_ssim(melpairedloader, same_name=
        same_name)
    out.update(metric_psnr_ssim)
    print('Getting Mel features...')
    featuresdict_2 = self.get_featuresdict(resultloader)
    featuresdict_1 = self.get_featuresdict(outputloader)
    print('Calculating KL divergence...')
    metric_kl, kl_ref, paths_1 = calculate_kl(featuresdict_1,
        featuresdict_2, 'logits', same_name)
    out.update(metric_kl)
    print('Calculating inception score...')
    metric_isc = calculate_isc(featuresdict_1, feat_layer_name='logits',
        splits=10, samples_shuffle=True, rng_seed=2020)
    out.update(metric_isc)
    print('Calculating kernel inception distance...')
    metric_kid = calculate_kid(featuresdict_1, featuresdict_2,
        feat_layer_name='2048', degree=3, gamma=None, subsets=100,
        subset_size=len(pairedtextdataset), coef0=1, rng_seed=2020)
    out.update(metric_kid)
    print('Calculating Frechet distance...')
    metric_fid = calculate_fid(featuresdict_1, featuresdict_2,
        feat_layer_name='2048')
    out.update(metric_fid)
    keys_list = ['frechet_distance', 'frechet_audio_distance', 'lsd',
        'psnr', 'kullback_leibler_divergence_sigmoid',
        'kullback_leibler_divergence_softmax', 'ssim', 'ssim_stft',
        'inception_score_mean', 'inception_score_std',
        'kernel_inception_distance_mean', 'kernel_inception_distance_std',
        'gt_text_clap_score', 'gen_text_clap_score', 'gen_gt_clap_score']
    result = {}
    for key in keys_list:
        result[key] = round(out.get(key, float('nan')), 4)
    json_path = generate_files_path + '_evaluation_results.json'
    write_json(result, json_path)
    return result
