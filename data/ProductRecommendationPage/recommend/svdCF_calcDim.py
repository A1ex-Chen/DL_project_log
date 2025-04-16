def calcDim(sigma):
    sigma2 = sigma ** 2
    for i in range(len(sigma)):
        if sum(sigma2[:i]) / sum(sigma2) > 0.9:
            return i
