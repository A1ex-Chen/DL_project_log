def log_results(accelerator, logger, epoch, completed_steps, lr, train_loss,
    val_loss, best_eval_loss, output_dir, with_tracking):
    save_checkpoint = False
    if accelerator.is_main_process:
        result = {}
        result['epoch'] = epoch,
        result['step'] = completed_steps
        result['lr'] = lr
        if len(val_loss) == 4:
            result['loss_wrt_gt'] = round(val_loss[0], 6)
            result['loss_wrt_teacher'] = round(val_loss[1], 6)
            result['consistency_loss'] = round(val_loss[2], 6)
            result['teacher_loss'] = round(val_loss[3], 6)
            result_string = (
                f'Epoch: {epoch}, Val loss wrt teacher: {val_loss[1]:.4f}, ')
            loss_to_track = result['loss_wrt_teacher']
        else:
            result['validation_loss'] = round(val_loss[0], 6)
            result_string = f'Epoch: {epoch}, Val loss: {val_loss[0]:.4f}, '
            loss_to_track = result['validation_loss']
        if train_loss is not None:
            result['train_loss'] = round(train_loss, 6)
        wandb.log(result)
        if train_loss is not None:
            result_string += f'Training loss: {train_loss:.4f}\n'
        logger.info(result_string)
        with open(f'{output_dir}/summary.jsonl', 'a') as f:
            f.write(json.dumps(result) + '\n\n')
        logger.info(result)
        if loss_to_track < best_eval_loss:
            best_eval_loss = loss_to_track
            save_checkpoint = True
    if with_tracking:
        accelerator.log(result, step=completed_steps)
    return save_checkpoint, best_eval_loss
