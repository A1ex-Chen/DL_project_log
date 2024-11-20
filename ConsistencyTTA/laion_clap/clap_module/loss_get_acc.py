def get_acc(pred, target):
    pred = torch.argmax(pred, 1).numpy()
    target = torch.argmax(target, 1).numpy()
    return accuracy_score(target, pred)
