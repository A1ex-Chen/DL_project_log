def train(train_loader, model, criterion, optimizer, scaler, epoch,
    lr_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':7.3f')
    mem = AverageMeter('Mem (GB)', ':4.0f')
    metric_names = models.get_metric_names(args.model)
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in
        metric_names])
    progress = ProgressMeter(iters_per_epoch, [batch_time, data_time, mem,
        *metrics.values()], prefix='Epoch: [{}]'.format(epoch))
    model.train()
    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq
        data_time.update(time.time() - end)
        it = iters_per_epoch * epoch + optim_iter
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]
        pc = inputs[3]
        texts = inputs[2]
        object_name = inputs[1]
        image = inputs[4]
        inputs = [pc, texts, image]
        inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs
            ]
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(*inputs)
            loss_dict = criterion(object_name, outputs)
            loss = loss_dict['loss']
            loss /= args.update_freq
        if not math.isfinite(loss.item()):
            print('Loss is {}, stopping training'.format(loss.item()))
            sys.exit(1)
        scaler.scale(loss).backward()
        if (data_iter + 1) % args.update_freq != 0:
            continue
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)
        utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        logit_scale = utils.get_model(model).logit_scale.exp().item()
        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()
        mem.update(torch.cuda.max_memory_allocated() // 1000000000.0)
        if optim_iter % args.print_freq == 0:
            if utils.is_main_process() and args.wandb:
                wandb.log({**{k: v.item() for k, v in loss_dict.items()},
                    'scaler': scaler.get_scale(), 'logit': logit_scale})
            progress.display(optim_iter)
    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()}, 'lr': optimizer.
        param_groups[0]['lr'], 'logit_scale': logit_scale}
