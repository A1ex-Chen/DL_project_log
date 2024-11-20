def get_mix_lambda(mixup_alpha, batch_size):
    mixup_lambdas = [np.random.beta(mixup_alpha, mixup_alpha, 1)[0] for _ in
        range(batch_size)]
    return np.array(mixup_lambdas).astype(np.float32)
