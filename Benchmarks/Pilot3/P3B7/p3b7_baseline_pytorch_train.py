def train(model, loader, optimizer, device, epoch):
    accmeter = AccuracyMeter(TASKS, loader)
    total_loss = 0
    for idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        data, target = data.to(device), to_device(target, device)
        logits = model(data)
        _ = TRAIN_F1_MICRO.f1(to_device(logits, 'cpu'), to_device(target,
            'cpu'))
        _ = TRAIN_F1_MACRO.f1(to_device(logits, 'cpu'), to_device(target,
            'cpu'))
        loss = model.loss_value(logits, target, reduce='mean')
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        accmeter.update(logits, target)
    avg_loss = total_loss / len(loader.dataset)
    accmeter.update_accuracy()
    print(f'\nEpoch {epoch} Training Accuracy:')
    accmeter.print_task_accuracies()
    accmeter.reset()
    return avg_loss
