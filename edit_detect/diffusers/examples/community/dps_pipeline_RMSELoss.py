def RMSELoss(yhat, y):
    return torch.sqrt(torch.sum((yhat - y) ** 2))
