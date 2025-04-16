def train(args, epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader, file=sys.stdout)
    criterion = nn.CrossEntropyLoss()
    for i, (top, bottom, label) in enumerate(loader):
        model.zero_grad()
        top = top.to(device)
        if args.hier == 'top':
            target = top
            out, _ = model(top)
        elif args.hier == 'bottom':
            bottom = bottom.to(device)
            target = bottom
            out, _ = model(bottom, condition=top)
        loss = criterion(out, target)
        loss.backward()
        if scheduler is not None:
            scheduler.step()
        optimizer.step()
        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()
        lr = optimizer.param_groups[0]['lr']
        loader.set_description(
            f'epoch: {epoch + 1}; loss: {loss.item():.5f}; acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
