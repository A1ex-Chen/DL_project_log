def preds2score(preds, splits=10):
    scores = []
    for i in range(splits):
        part = preds[i * preds.shape[0] // splits:(i + 1) * preds.shape[0] //
            splits, :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0))
            )
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)
