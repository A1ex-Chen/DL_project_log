def train(trainloader, validloader, model, architecture, criterion,
    optimizer, lr, args, tasks, device, meter):
    valid_iter = iter(trainloader)
    for step, (data, target) in enumerate(trainloader):
        batch_size = data.size(0)
        model.train()
        data = data.to(device)
        for task, label in target.items():
            target[task] = target[task].to(device)
        x_search, target_search = next(valid_iter)
        x_search = x_search.to(device)
        for task, label in target_search.items():
            target_search[task] = target_search[task].to(device)
        architecture.step(data, target, x_search, target_search, lr,
            optimizer, unrolled=args.unrolled)
        logits = model(data)
        loss = darts.multitask_loss(target, logits, criterion, reduce='mean')
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        prec1 = darts.multitask_accuracy_topk(logits, target)
        meter.update_batch_loss(loss.item(), batch_size)
        meter.update_batch_accuracy(prec1, batch_size)
        if step % args.log_interval == 0:
            print(f'Step: {step} loss: {meter.loss_meter.avg:.4}')
    meter.update_epoch()
    meter.save(args.save_path)
