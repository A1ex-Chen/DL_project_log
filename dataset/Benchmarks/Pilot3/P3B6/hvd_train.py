def train(dataloader, sampler, model, optimizer, criterion, args, epoch):
    model.train()
    sampler.set_epoch(epoch)
    for idx, batch in enumerate(dataloader):
        train_loss = 0.0
        optimizer.zero_grad()
        input_ids = batch['tokens'].to(args.device)
        segment_ids = batch['seg_ids'].to(args.device)
        input_mask = batch['masks'].to(args.device)
        logits = model(input_ids, input_mask, segment_ids)
        labels = batch['label'].to(args.device)
        loss = criterion(logits.view(-1, args.num_classes), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.mean()
        if (idx + 1) % 100 == 0:
            train_loss = torch.tensor(train_loss)
            avg_loss = hvd.allreduce(train_loss, name='avg_loss').item()
            if hvd.rank() == 0:
                print(f'epoch: {epoch}, batch: {idx}, loss: {train_loss}')
