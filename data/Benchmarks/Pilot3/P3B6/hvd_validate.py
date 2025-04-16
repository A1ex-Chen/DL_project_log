def validate(dataloader, model, args, epoch):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            input_ids = batch['tokens'].to(args.device)
            segment_ids = batch['seg_ids'].to(args.device)
            input_mask = batch['masks'].to(args.device)
            logits = model(input_ids, input_mask, segment_ids)
            logits = torch.nn.Sigmoid()(logits)
            logits = logits.view(-1, args.num_classes).cpu().data.numpy()
            preds.append(np.rint(logits))
            labels.append(batch['label'].data.numpy())
    preds = np.concatenate(preds, 0)
    labels = np.concatenate(labels, 0)
    preds = torch.tensor(preds)
    preds_all = hvd.allgather(preds, name='val_preds_all').cpu().data.numpy()
    labels = torch.tensor(labels)
    labels_all = hvd.allgather(labels, name='val_labels_all').cpu().data.numpy(
        )
    valid_f1 = f1_score(labels_all.flatten(), preds_all.flatten())
    if hvd.rank() == 0:
        print(f'epoch: {epoch}, validation F1: {valid_f1}')
