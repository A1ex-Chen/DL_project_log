def get_map(pred, target):
    pred = torch.sigmoid(pred).numpy()
    target = target.numpy()
    return np.mean(average_precision_score(target, pred, average=None))
