def compute_loss_stateless_model(params, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)
    predictions = fmodel(params, batch)
    loss = criterion(predictions, targets)
    return loss
