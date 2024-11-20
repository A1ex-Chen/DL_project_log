def train(tr_iter, va_iter, model, para_model, model_config, optimizer,
    optimizer_sparse, scheduler, scheduler_sparse, scaler, vocab, epoch,
    last_batch, last_iter, train_step, best_val_loss, meters,
    timeout_handler, device, args):
    model.train()
    train_loss = 0
    target_tokens = 0
    log_step = 0
    log_start_time = time.time()
    mems = [None for _ in range(args.batch_chunk)]
    if args.varlen:
        train_iter = tr_iter.get_varlen_iter(start=last_iter)
    else:
        train_iter = tr_iter.get_fixlen_iter(start=last_iter)
    for batch, (data, target, seq_len, _) in enumerate(train_iter, start=
        last_batch + 1):
        log_step += 1
        target_tokens += target.numel()
        for param in model.parameters():
            param.grad = None
        data_chunks = torch.chunk(data, args.batch_chunk, 1)
        target_chunks = torch.chunk(target, args.batch_chunk, 1)
        for i in range(args.batch_chunk):
            if i < args.batch_chunk - 1 and isinstance(para_model,
                DistributedDataParallel):
                with para_model.no_sync():
                    train_loss_chunk = train_iteration(para_model, i, mems,
                        data_chunks, target_chunks, scaler, optimizer,
                        device, True, args)
            else:
                train_loss_chunk = train_iteration(para_model, i, mems,
                    data_chunks, target_chunks, scaler, optimizer, device, 
                    False, args)
            train_loss += train_loss_chunk
        if args.fp16:
            if args.amp == 'pytorch':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            elif args.amp == 'apex':
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                    args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if args.fp16 and args.amp == 'pytorch':
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            if optimizer_sparse:
                optimizer_sparse.step()
        train_step += 1
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
                if optimizer_sparse:
                    optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            elif args.scheduler == 'cosine':
                scheduler.step(train_step - args.warmup_step)
                if scheduler_sparse:
                    scheduler_sparse.step(train_step - args.warmup_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)
            if scheduler_sparse:
                scheduler_sparse.step(train_step)
        if train_step % args.log_interval == 0:
            cur_loss = train_loss / log_step
            cur_loss = utils.distributed.all_reduce_item(cur_loss, op='mean')
            train_loss = 0
            elapsed = time.time() - log_start_time
            avg_elapsed = elapsed / log_step
            avg_elapsed = utils.distributed.all_reduce_item(avg_elapsed, op
                ='max')
            log_start_time = time.time()
            log_step = 0
            lr = optimizer.param_groups[0]['lr']
            throughput = target_tokens / elapsed
            throughput = utils.distributed.all_reduce_item(throughput, op='sum'
                )
            meters['train_throughput'].update(throughput)
            target_tokens = 0
            log_str = (
                '| epoch {:3d} step {:>8d} | batches {:>6d} / {:d} | lr {:.3e} | ms/batch {:5.1f} | tok/s {:7.0f} | loss {:5.2f}'
                .format(epoch, train_step, batch, tr_iter.n_batch, lr, 
                avg_elapsed * 1000, throughput, cur_loss))
            dllogger_data = {'epoch': epoch, 'train_batch': batch + 1, 'lr':
                lr, 'train_time/batch': avg_elapsed * 1000,
                'train_throughput': throughput, 'train_loss': cur_loss}
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(cur_loss / math.log(2))
                dllogger_data['train_bits_per_character'
                    ] = cur_loss / math.log(2)
            else:
                log_str += ' | ppl {:9.2f}'.format(math.exp(cur_loss))
                dllogger_data['train_perplexity'] = math.exp(cur_loss)
            logging.info(log_str)
            dllogger.log(step=tuple([train_step]), data=dllogger_data)
        do_periodic_eval = train_step % args.eval_interval == 0
        is_final_step = train_step == args.max_step
        interrupted = timeout_handler.interrupted
        if (do_periodic_eval or is_final_step or interrupted
            ) and not args.no_eval:
            eval_start_time = time.time()
            val_loss = evaluate(va_iter, model, args)
            val_loss = utils.distributed.all_reduce_item(val_loss, op='mean')
            logging.info('-' * 100)
            log_str = (
                '| Eval {:3d} at step {:>8d} | time: {:5.2f}s | valid loss {:5.2f}'
                .format(train_step // args.eval_interval, train_step, time.
                time() - eval_start_time, val_loss))
            dllogger_data = {'valid_elapsed': time.time() - eval_start_time,
                'valid_loss': val_loss}
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(val_loss / math.log(2))
                dllogger_data['valid_bits_per_character'
                    ] = val_loss / math.log(2)
            else:
                log_str += ' | valid ppl {:9.3f}'.format(math.exp(val_loss))
                dllogger_data['valid_perplexity'] = math.exp(val_loss)
            logging.info(log_str)
            logging.info('-' * 100)
            dllogger.log(step=tuple([train_step]), data=dllogger_data)
            last_iter = tr_iter.last_iter
            is_best = False
            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
                is_best = True
            if not args.debug:
                save_checkpoint(args, model, model_config, optimizer,
                    scheduler, scaler, vocab, epoch, batch, last_iter,
                    train_step, best_val_loss, is_best, args.work_dir)
            if args.scheduler == 'dev_perf':
                scheduler.step(val_loss)
                if scheduler_sparse:
                    scheduler_sparse.step(val_loss)
            log_start_time += time.time() - eval_start_time
        if interrupted:
            logging.info(f'Received SIGTERM, exiting')
            sys.exit(0)
        if is_final_step:
            break
    return train_step, best_val_loss
