def calculate_kid(featuresdict_1, featuresdict_2, subsets, subset_size,
    degree, gamma, coef0, rng_seed, feat_layer_name):
    features_1 = featuresdict_1[feat_layer_name]
    features_2 = featuresdict_2[feat_layer_name]
    assert torch.is_tensor(features_1) and features_1.dim() == 2
    assert torch.is_tensor(features_2) and features_2.dim() == 2
    assert features_1.shape[1] == features_2.shape[1]
    if subset_size > len(features_2):
        print(
            f'WARNING: subset size ({subset_size}) is larger than feature length ({len(features_2)}). '
            , 'Using', len(features_2), 'for both datasets')
        subset_size = len(features_2)
    if subset_size > len(features_1):
        print(
            f'WARNING: subset size ({subset_size}) is larger than feature length ({len(features_1)}). '
            , 'Using', len(features_1), 'for both datasets')
        subset_size = len(features_1)
    features_1 = features_1.cpu().numpy()
    features_2 = features_2.cpu().numpy()
    mmds = np.zeros(subsets)
    rng = np.random.RandomState(rng_seed)
    for i in tqdm(range(subsets), leave=False, unit='subsets', desc=
        'Computing Kernel Inception Distance'):
        f1 = features_1[rng.choice(len(features_1), subset_size, replace=False)
            ]
        f2 = features_2[rng.choice(len(features_2), subset_size, replace=False)
            ]
        o = polynomial_mmd(f1, f2, degree, gamma, coef0)
        mmds[i] = o
    return {'kernel_inception_distance_mean': float(np.mean(mmds)),
        'kernel_inception_distance_std': float(np.std(mmds))}
