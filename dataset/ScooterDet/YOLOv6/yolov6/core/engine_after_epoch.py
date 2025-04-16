def after_epoch(self):
    lrs_of_this_epoch = [x['lr'] for x in self.optimizer.param_groups]
    self.scheduler.step()
    if self.main_process:
        self.ema.update_attr(self.model, include=['nc', 'names', 'stride'])
        remaining_epochs = self.max_epoch - 1 - self.epoch
        eval_interval = (self.args.eval_interval if remaining_epochs >=
            self.args.heavy_eval_range else min(3, self.args.eval_interval))
        is_val_epoch = (remaining_epochs == 0 or not self.args.
            eval_final_only and (self.epoch + 1) % eval_interval == 0)
        if is_val_epoch:
            self.eval_model()
            self.ap = self.evaluate_results[1]
            self.best_ap = max(self.ap, self.best_ap)
        ckpt = {'model': deepcopy(de_parallel(self.model)).half(), 'ema':
            deepcopy(self.ema.ema).half(), 'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(), 'scheduler': self.
            scheduler.state_dict(), 'epoch': self.epoch, 'results': self.
            evaluate_results}
        save_ckpt_dir = osp.join(self.save_dir, 'weights')
        save_checkpoint(ckpt, is_val_epoch and self.ap == self.best_ap,
            save_ckpt_dir, model_name='last_ckpt')
        if self.epoch >= self.max_epoch - self.args.save_ckpt_on_last_n_epoch:
            save_checkpoint(ckpt, False, save_ckpt_dir, model_name=
                f'{self.epoch}_ckpt')
        if self.epoch >= self.max_epoch - self.args.stop_aug_last_n_epoch:
            if self.best_stop_strong_aug_ap < self.ap:
                self.best_stop_strong_aug_ap = max(self.ap, self.
                    best_stop_strong_aug_ap)
                save_checkpoint(ckpt, False, save_ckpt_dir, model_name=
                    'best_stop_aug_ckpt')
        del ckpt
        self.evaluate_results = list(self.evaluate_results)
        write_tblog(self.tblogger, self.epoch, self.evaluate_results,
            lrs_of_this_epoch, self.mean_loss)
        write_tbimg(self.tblogger, self.vis_imgs_list, self.epoch, type='val')
