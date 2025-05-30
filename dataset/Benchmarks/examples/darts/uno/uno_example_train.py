def train(trainloader, model, architecture, criterion, optimizer, scheduler,
    args, tasks, meter, genotype, genotype_store, device):
    valid_iter = iter(trainloader)
    min_accuracy = 0.0
    for step, (data, target) in enumerate(trainloader):
        batch_size = data.size(0)
        model.train()
        data = darts.to_device(data, device)
        target = darts.to_device(target, device)
        x_search, target_search = next(valid_iter)
        x_search = darts.to_device(x_search, device)
        target_search = darts.to_device(target_search, device)
        lr = scheduler.get_lr()[0]
        architecture.step(data, target, x_search, target_search, lr,
            optimizer, unrolled=False)
        logits = model(data)
        loss = darts.multitask_loss(target, logits, criterion, reduce='mean')
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        prec1 = darts.multitask_accuracy_topk(logits, target, topk=(1,))
        meter.update_batch_loss(loss.item(), batch_size)
        meter.update_batch_accuracy(prec1, batch_size)
        accuracy_avg = meter.acc_meter.get_avg_accuracy('response')
        if accuracy_avg > min_accuracy:
            genotype_store.save_genotype(genotype)
            min_accuracy = accuracy_avg
        if step % args.log_interval == 0:
            logger.info(f'Step: {step} loss: {meter.loss_meter.avg:.4}')
    meter.update_epoch()
    meter.save(args.save_path)
