def infer(validloader, model, criterion, args, tasks, device, meter):
    model.eval()
    with torch.no_grad():
        for step, (data, target) in enumerate(validloader):
            data = data.to(device)
            for task, label in target.items():
                target[task] = target[task].to(device)
            batch_size = data.size(0)
            logits = model(data)
            loss = darts.multitask_loss(target, logits, criterion, reduce=
                'mean')
            prec1 = darts.multitask_accuracy_topk(logits, target)
            meter.update_batch_loss(loss.item(), batch_size)
            meter.update_batch_accuracy(prec1, batch_size)
            if step % args.log_interval == 0:
                print(f'>> Validation: {step} loss: {meter.loss_meter.avg:.4}')
    meter.update_epoch()
    meter.save(args.save_path)
    return meter.loss_meter.avg
