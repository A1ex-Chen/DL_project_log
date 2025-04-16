def validate(model, loader, loss_fn, args, device=None, amp_autocast=
    suppress, log_suffix=''):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available(
            ) else torch.device('cpu')
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    model.eval()
    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.to(device)
                target = target.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            with amp_autocast():
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor
                        ).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]
                loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data
            if device.type == 'cuda':
                torch.cuda.synchronize()
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (last_batch or batch_idx % args.
                log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})'
                    )
            if args.dryrun:
                break
    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), (
        'top5', top5_m.avg)])
    return metrics
