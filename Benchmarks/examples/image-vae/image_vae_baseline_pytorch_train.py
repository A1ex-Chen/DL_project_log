def train(epoch, args):
    print('Epoch {}: batch_size {}'.format(epoch, get_batch_size(epoch, args)))
    model.train()
    loss_meter = AverageMeter()
    for batch_idx, (_, data, _) in enumerate(train_loader_food):
        data = data.float().cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar, _ = model(data)
        loss2 = loss_picture(recon_batch, data, mu, logvar, epoch)
        loss2 = torch.sum(loss2)
        loss_meter.update(loss2.item(), int(recon_batch.shape[0]))
        loss2.backward()
        clip_gradient(optimizer, grad_clip=args.grad_clip)
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {}'.
                format(epoch, batch_idx * len(data), len(train_loader_food.
                dataset), 100.0 * batch_idx / len(train_loader_food),
                loss_meter.avg, datetime.datetime.now()))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_meter.avg))
    return loss_meter.avg
