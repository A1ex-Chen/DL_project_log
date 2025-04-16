def validate(dataloader, model, args, device, epoch):
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            input_ids = batch['tokens'].to(device)
            labels = batch['label'].to(args.device)
            output = model(input_ids, labels=labels)
            print(f'epoch: {epoch}, batch: {idx}, valid loss: {output.loss}')
