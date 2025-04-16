def evaluate(model, loader, device):
    accmeter = AccuracyMeter(TASKS, loader)
    loss = 0
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(loader):
            data, target = data.to(device), to_device(target, device)
            logits = model(data)
            _ = VALID_F1_MICRO.f1(to_device(logits, 'cpu'), to_device(
                target, 'cpu'))
            _ = VALID_F1_MACRO.f1(to_device(logits, 'cpu'), to_device(
                target, 'cpu'))
            loss += model.loss_value(logits, target, reduce='mean').item()
            accmeter.update(logits, target)
    accmeter.update_accuracy()
    print('Validation accuracy:')
    accmeter.print_task_accuracies()
    loss /= len(loader.dataset)
    return loss
