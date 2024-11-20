def compute_accuracy(self, y_hat, y):
    with torch.no_grad():
        y_pred = y_hat >= 0.5
        y_pred_f = y_pred.float()
        num_correct = tsum(y_pred_f == y)
        denom = float(y.size()[0])
        accuracy = torch.div(num_correct, denom)
    return accuracy
