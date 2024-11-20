def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args,
    tb_writer=None, extra_suffix=''):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    model.train()
    loss = LPLoss(args.lp_loss)
    dataloader, sampler = data['train'].dataloader, data['train'].sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    if args.dataset_type == 'toy':
        dataloader.dataset.generate_queue()
    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        if isinstance(scheduler, dict):
            for s in scheduler.values():
                s(step)
        else:
            scheduler(step)
        audio = batch
        class_label = batch['class_label']
        class_label = class_label.to(device=device, non_blocking=True)
        if args.mixup:
            mix_lambda = torch.from_numpy(get_mix_lambda(0.5, len(audio[
                'waveform']))).to(device)
            class_label = do_mixup(class_label, mix_lambda)
        else:
            mix_lambda = None
        data_time_m.update(time.time() - end)
        if isinstance(optimizer, dict):
            for o_ in optimizer.values():
                o_.zero_grad()
        else:
            optimizer.zero_grad()
        with autocast():
            pred = model(audio, mix_lambda=mix_lambda, device=device)
            total_loss = loss(pred, class_label)
        if isinstance(optimizer, dict):
            if scaler is not None:
                scaler.scale(total_loss).backward()
                for o_ in optimizer.values():
                    if args.horovod:
                        o_.synchronize()
                        scaler.unscale_(o_)
                        with o_.skip_synchronize():
                            scaler.step(o_)
                    else:
                        scaler.step(o_)
                scaler.update()
            else:
                total_loss.backward()
                for o_ in optimizer.values():
                    o_.step()
        elif scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        with torch.no_grad():
            unwrap_model(model).clap_model.logit_scale_a.clamp_(0, math.log
                (100))
            unwrap_model(model).clap_model.logit_scale_t.clamp_(0, math.log
                (100))
        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count ==
            num_batches_per_epoch):
            if isinstance(audio, dict):
                batch_size = len(audio['waveform'])
            else:
                batch_size = len(audio)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch
            loss_m.update(total_loss.item(), batch_size)
            if isinstance(optimizer, dict):
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) Data (t): {data_time_m.avg:.3f} Batch (t): {batch_time_m.avg:.3f} LR: {[o_.param_groups[0]['lr'] for o_ in optimizer.values()]}"
                    )
                log_data = {'loss': loss_m.val, 'data_time': data_time_m.
                    val, 'batch_time': batch_time_m.val, 'lr': [o_.
                    param_groups[0]['lr'] for o_ in optimizer.values()]}
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) Data (t): {data_time_m.avg:.3f} Batch (t): {batch_time_m.avg:.3f} LR: {optimizer.param_groups[0]['lr']:5f} "
                    )
                log_data = {'loss': loss_m.val, 'data_time': data_time_m.
                    val, 'batch_time': batch_time_m.val, 'lr': optimizer.
                    param_groups[0]['lr']}
            for name, val in log_data.items():
                name = f'train{extra_suffix}/{name}'
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})
            batch_time_m.reset()
            data_time_m.reset()
