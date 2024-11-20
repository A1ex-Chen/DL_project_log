def calculate_fid(act1, act2):
    mu1 = act1.mean(axis=0)
    sigma1 = np.cov(act1, rowvar=False)
    mu2 = np.mean(act2, axis=0)
    sigma2 = np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2)
    mual = sigma1.dot(sigma2)
    covmean = linalg.sqrtm(mual)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    trace = np.trace(sigma1 + sigma2 - 2.0 * covmean)
    fid = ssdiff + trace
    return fid
