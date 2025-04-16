def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args, device=
    None, lr_scheduler=None, saver=None, output_dir=None, amp_autocast=
    suppress, loss_scaler=None, model_ema=None, mixup_fn=None, model_kd=None):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available(
            ) else torch.device('cpu')
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False
    second_order = hasattr(optimizer, 'is_second_order'
        ) and optimizer.is_second_order
    has_no_sync = hasattr(model, 'no_sync')
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    model.train()
    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps
    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps
        if not args.prefetcher:
            input, target = input.to(device), target.to(device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            with amp_autocast():
                output = model(input)
                loss = loss_fn(output, target)
                if model_kd is not None:
                    if not args.use_kd_loss_only:
                        loss += args.alpha_kd * compute_kd_loss(input,
                            output, model, model_kd)
                    else:
                        loss = compute_kd_loss(input, output, model, model_kd)
            if accum_steps > 1:
                loss /= accum_steps
            return loss

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(_loss, optimizer, clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode, parameters=model_parameters(
                    model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order, need_update=need_update)
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(model_parameters(model,
                            exclude_head='agc' in args.clip_mode), value=
                            args.clip_grad, mode=args.clip_mode)
                    optimizer.step()
        if has_no_sync and not need_update:
            with model.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            _backward(loss)
        if not args.distributed:
            losses_m.update(loss.item() * accum_steps, input.size(0))
        update_sample_count += input.size(0)
        if not need_update:
            data_start_time = time.time()
            continue
        num_updates += 1
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model)
        if args.synchronize_step and device.type == 'cuda':
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now
        if update_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item() * accum_steps, input.
                    size(0))
                update_sample_count *= args.world_size
            if utils.is_primary(args):
                _logger.info(
                    f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} ({100.0 * update_idx / (updates_per_epoch - 1):>3.0f}%)]  Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  ({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  LR: {lr:.3e}  Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                    )
                if args.save_images and output_dir:
                    torchvision.utils.save_image(input, os.path.join(
                        output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0, normalize=True)
        if args.dryrun:
            break
        if saver is not None and args.recovery_interval and (update_idx + 1
            ) % args.recovery_interval == 0:
            saver.save_recovery(epoch, batch_idx=update_idx)
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=
                losses_m.avg)
        update_sample_count = 0
        data_start_time = time.time()
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    return OrderedDict([('loss', losses_m.avg)])
