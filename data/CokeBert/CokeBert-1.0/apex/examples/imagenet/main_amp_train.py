def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()
    prefetcher = data_prefetcher(train_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        i += 1
        if args.prof >= 0 and i == args.prof:
            print('Profiling begun at iteration {}'.format(i))
            torch.cuda.cudart().cudaProfilerStart()
        if args.prof >= 0:
            torch.cuda.nvtx.range_push('Body of iteration {}'.format(i))
        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        if args.prof >= 0:
            torch.cuda.nvtx.range_push('forward')
        output = model(input)
        if args.prof >= 0:
            torch.cuda.nvtx.range_pop()
        loss = criterion(output, target)
        optimizer.zero_grad()
        if args.prof >= 0:
            torch.cuda.nvtx.range_push('backward')
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if args.prof >= 0:
            torch.cuda.nvtx.range_pop()
        if args.prof >= 0:
            torch.cuda.nvtx.range_push('optimizer.step()')
        optimizer.step()
        if args.prof >= 0:
            torch.cuda.nvtx.range_pop()
        if i % args.print_freq == 0:
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
            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()
            if args.local_rank == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tSpeed {3:.3f} ({4:.3f})\tLoss {loss.val:.10f} ({loss.avg:.4f})\tPrec@1 {top1.val:.3f} ({top1.avg:.3f})\tPrec@5 {top5.val:.3f} ({top5.avg:.3f})'
                    .format(epoch, i, len(train_loader), args.world_size *
                    args.batch_size / batch_time.val, args.world_size *
                    args.batch_size / batch_time.avg, batch_time=batch_time,
                    loss=losses, top1=top1, top5=top5))
        if args.prof >= 0:
            torch.cuda.nvtx.range_push('prefetcher.next()')
        input, target = prefetcher.next()
        if args.prof >= 0:
            torch.cuda.nvtx.range_pop()
        if args.prof >= 0:
            torch.cuda.nvtx.range_pop()
        if args.prof >= 0 and i == args.prof + 10:
            print('Profiling ended at iteration {}'.format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()
