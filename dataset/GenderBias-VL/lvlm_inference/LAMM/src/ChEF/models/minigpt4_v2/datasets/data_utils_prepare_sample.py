def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)
    return samples
