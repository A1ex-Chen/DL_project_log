def validate(validation_batches, model, criterion, device, n_val_images,
    logs, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(n_val_images, [batch_time, losses, top1, top5],
        prefix='Test: ')
    model.eval()
    with torch.no_grad():
        end = time.time()
        examples_done = 0
        for i, validation_batch in enumerate(validation_batches):
            images = torch.from_numpy(validation_batch[0].numpy())
            target = torch.from_numpy(validation_batch[1].numpy())
            images = images.to(device)
            target = target.to(device)
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            examples_done += len(target)
            if i % args.print_freq == 0:
                progress.display(examples_done)
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=
            top1, top5=top5), flush=True)
        logs['acc_val_top-1'] = top1.avg.cpu().numpy().item()
        logs['acc_val_top-5'] = top5.avg.cpu().numpy().item()
        logs['loss_val'] = losses.avg
    return logs
