def train(dataloader, model, optimizer, criterion, args, epoch):
    model.train()
    for idx, batch in enumerate(dataloader):
        train_loss = 0.0
        optimizer.zero_grad()
        input_ids = batch['tokens'].to(args.device)
        segment_ids = batch['seg_ids'].to(args.device)
        input_mask = batch['masks'].to(args.device)
        labels = batch['label'].to(args.device)
        output = model(input_ids, labels=labels)
        output.loss.backward()
        optimizer.step()
        print(f'epoch: {epoch}, batch: {idx}, train loss: {output.loss}')
