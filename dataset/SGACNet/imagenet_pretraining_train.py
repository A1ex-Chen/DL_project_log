def train(train_batches, model, criterion, optimizer, epoch, device,
    n_train_images, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(n_train_images, [batch_time, data_time, losses,
        top1, top5], prefix='Epoch: [{}]'.format(epoch))
    model.train()
    end = time.time()
    for i, train_batch in enumerate(train_batches):
        images = torch.from_numpy(train_batch[0].numpy())
        target = torch.from_numpy(train_batch[1].numpy())
        current_batch_size = len(target)
        if current_batch_size < args.batch_size:
            break
        data_time.update(time.time() - end)
        images = images.to(device)
        target = target.to(device)
        output = model(images)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display((i + 1) * args.batch_size)
    logs = dict()
    logs['acc_train_top-1'] = top1.avg.cpu().numpy().item()
    logs['acc_train_top-5'] = top5.avg.cpu().numpy().item()
    logs['loss_train'] = losses.avg
    return logs
