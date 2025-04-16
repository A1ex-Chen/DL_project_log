def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()
    run_info_dict = {'Iteration': [], 'Loss': [], 'Speed': []}
    prefetcher = data_prefetcher(train_loader)
    input, target = prefetcher.next()
    i = -1
    while input is not None:
        i += 1
        if args.prof:
            if i > 10:
                break
        data_time.update(time.time() - end)
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
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.cuda.nvtx.range_push('step')
        optimizer.step()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        input, target = prefetcher.next()
        if i % args.print_freq == 0 and i > 1:
            if args.local_rank == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tSpeed {3:.3f} ({4:.3f})\tData {data_time.val:.3f} ({data_time.avg:.3f})\tLoss {loss.val:.10f} ({loss.avg:.4f})\tPrec@1 {top1.val:.3f} ({top1.avg:.3f})\tPrec@5 {top5.val:.3f} ({top5.avg:.3f})'
                    .format(epoch, i, len(train_loader), args.world_size *
                    args.batch_size / batch_time.val, args.world_size *
                    args.batch_size / batch_time.avg, batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
            run_info_dict['Iteration'].append(i)
            run_info_dict['Loss'].append(losses.val)
            run_info_dict['Speed'].append(args.world_size * args.batch_size /
                batch_time.val)
            if len(run_info_dict['Loss']) == args.prints_to_process:
                if args.local_rank == 0:
                    torch.save(run_info_dict, str(args.has_ext) + '_' + str
                        (args.opt_level) + '_' + str(args.loss_scale) + '_' +
                        str(args.keep_batchnorm_fp32) + '_' + str(args.
                        fused_adam))
                quit()
