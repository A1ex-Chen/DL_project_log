def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args,
    tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    model.train()
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.
        gather_with_grad, cache_labels=True, rank=args.rank, world_size=
        args.world_size, use_horovod=args.horovod, mlp_loss=args.
        clap_mlploss, weight_loss_kappa=args.kappa)
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
        audios = batch
        texts = batch['text']
        data_time_m.update(time.time() - end)
        if isinstance(optimizer, dict):
            for o_ in optimizer.values():
                o_.zero_grad()
        else:
            optimizer.zero_grad()
        with autocast():
            (audio_features, text_features, audio_features_mlp,
                text_features_mlp, logit_scale_a, logit_scale_t) = model(audios
                , texts, device)
            if args.clap_mlploss:
                total_loss = loss(audio_features=audio_features,
                    text_features=text_features, logit_scale_a=
                    logit_scale_a, logit_scale_t=logit_scale_t,
                    audio_features_mlp=audio_features_mlp,
                    text_features_mlp=text_features_mlp)
            else:
                total_loss = loss(audio_features=audio_features,
                    text_features=text_features, logit_scale_a=logit_scale_a)
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
            unwrap_model(model).logit_scale_a.clamp_(0, math.log(100))
            if args.clap_mlploss:
                unwrap_model(model).logit_scale_t.clamp_(0, math.log(100))
        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count ==
            num_batches_per_epoch):
            if isinstance(audios, dict):
                batch_size = len(audios['waveform'])
            else:
                batch_size = len(audios)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar_a = logit_scale_a.item()
            logit_scale_scalar_t = logit_scale_t.item()
            if isinstance(optimizer, dict):
                if args.clap_mlploss:
                    logging.info(
                        f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) Data (t): {data_time_m.avg:.3f} Batch (t): {batch_time_m.avg:.3f} LR: {[o_.param_groups[0]['lr'] for o_ in optimizer.values()]} Logit Scale Audio: {logit_scale_scalar_a:.3f}Logit Scale Text: {logit_scale_scalar_t:.3f}"
                        )
                    log_data = {'loss': loss_m.val, 'data_time':
                        data_time_m.val, 'batch_time': batch_time_m.val,
                        'scale_audio': logit_scale_scalar_a, 'scale_text':
                        logit_scale_scalar_t, 'lr': [o_.param_groups[0][
                        'lr'] for o_ in optimizer.values()]}
                else:
                    logging.info(
                        f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) Data (t): {data_time_m.avg:.3f} Batch (t): {batch_time_m.avg:.3f} LR: {[o_.param_groups[0]['lr'] for o_ in optimizer.values()]} Logit Scale Audio: {logit_scale_scalar_a:.3f}"
                        )
                    log_data = {'loss': loss_m.val, 'data_time':
                        data_time_m.val, 'batch_time': batch_time_m.val,
                        'scale_audio': logit_scale_scalar_a, 'lr': [o_.
                        param_groups[0]['lr'] for o_ in optimizer.values()]}
            elif args.clap_mlploss:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) Data (t): {data_time_m.avg:.3f} Batch (t): {batch_time_m.avg:.3f} LR: {optimizer.param_groups[0]['lr']:5f} Logit Scale Audio: {logit_scale_scalar_a:.3f}Logit Scale Text: {logit_scale_scalar_t:.3f}"
                    )
                log_data = {'loss': loss_m.val, 'data_time': data_time_m.
                    val, 'batch_time': batch_time_m.val, 'scale_audio':
                    logit_scale_scalar_a, 'scale_text':
                    logit_scale_scalar_t, 'lr': optimizer.param_groups[0]['lr']
                    }
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) Data (t): {data_time_m.avg:.3f} Batch (t): {batch_time_m.avg:.3f} LR: {optimizer.param_groups[0]['lr']:5f} Logit Scale Audio: {logit_scale_scalar_a:.3f}"
                    )
                log_data = {'loss': loss_m.val, 'data_time': data_time_m.
                    val, 'batch_time': batch_time_m.val, 'scale_audio':
                    logit_scale_scalar_a, 'lr': optimizer.param_groups[0]['lr']
                    }
            for name, val in log_data.items():
                name = 'train/' + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})
            batch_time_m.reset()
            data_time_m.reset()
