def save_checkpoint(args, model, model_config, optimizer, scheduler, scaler,
    vocab, epoch, batch, last_iter, train_step, best_val_loss, is_best,
    work_dir):
    if args.fp16:
        if args.amp == 'pytorch':
            amp_state = scaler.state_dict()
        elif args.amp == 'apex':
            amp_state = amp.state_dict()
    else:
        amp_state = None
    state = {'args': args, 'model_config': model_config, 'model_state':
        model.state_dict(), 'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(), 'vocab': vocab,
        'amp_state': amp_state, 'epoch': epoch, 'batch': batch, 'last_iter':
        last_iter, 'train_step': train_step, 'best_val_loss': best_val_loss}
    last_chkpt_fname = 'checkpoint_last.pt'
    with utils.distributed.sync_workers() as rank:
        last_chkpt_path = os.path.join(work_dir, last_chkpt_fname)
        if rank == 0:
            logging.info(f'Saving checkpoint to {last_chkpt_path}')
            torch.save(state, last_chkpt_path)
            if is_best:
                best_chkpt_fname = 'checkpoint_best.pt'
                best_chkpt_path = os.path.join(work_dir, best_chkpt_fname)
                logging.info(f'Saving checkpoint to {best_chkpt_path}')
                shutil.copy(last_chkpt_path, best_chkpt_path)
            if args.save_all:
                step_chkpt_fname = f'checkpoint_{train_step}.pt'
                step_chkpt_path = os.path.join(work_dir, step_chkpt_fname)
                logging.info(f'Saving checkpoint to {step_chkpt_path}')
                shutil.copy(last_chkpt_path, step_chkpt_path)
