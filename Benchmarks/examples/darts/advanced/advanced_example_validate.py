def validate(validloader, model, criterion, args, tasks, meter, device):
    model.eval()
    with torch.no_grad():
        for step, (data, target) in enumerate(validloader):
            target = _wrap_target(target)
            data = darts.to_device(data, device)
            target = darts.to_device(target, device)
            batch_size = data.size(0)
            logits = model(data)
            loss = darts.multitask_loss(target, logits, criterion, reduce=
                'mean')
            prec1 = darts.multitask_accuracy_topk(logits, target, topk=(1,))
            meter.update_batch_loss(loss.item(), batch_size)
            meter.update_batch_accuracy(prec1, batch_size)
            if step % args.log_interval == 0:
                logger.info(
                    f'>> Validation: {step} loss: {meter.loss_meter.avg:.4}')
    meter.update_epoch()
    meter.save(args.save_path)
