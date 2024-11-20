def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    print('mu1 ', mu1.shape)
    print('mu2 ', mu2.shape)
    print('sigma1 ', sigma1.shape)
    print('sigma2 ', sigma2.shape)
    ssdiff = numpy.sum((mu1 - mu2) * 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if iscomplexobj(covmean):
        covmean = covmean.real
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
