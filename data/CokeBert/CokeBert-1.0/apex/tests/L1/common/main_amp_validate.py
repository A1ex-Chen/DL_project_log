def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    end = time.time()
    prefetcher = data_prefetcher(val_loader)
    input, target = prefetcher.next()
    i = -1
    while input is not None:
        i += 1
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data
        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.local_rank == 0 and i % args.print_freq == 0:
            print(
                'Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tSpeed {2:.3f} ({3:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})\tPrec@1 {top1.val:.3f} ({top1.avg:.3f})\tPrec@5 {top5.val:.3f} ({top5.avg:.3f})'
                .format(i, len(val_loader), args.world_size * args.
                batch_size / batch_time.val, args.world_size * args.
                batch_size / batch_time.avg, batch_time=batch_time, loss=
                losses, top1=top1, top5=top5))
        input, target = prefetcher.next()
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1,
        top5=top5))
    return top1.avg
