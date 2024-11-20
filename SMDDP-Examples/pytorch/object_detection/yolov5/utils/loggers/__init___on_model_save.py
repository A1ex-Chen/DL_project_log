def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
    if self.wandb:
        if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch
            ) and self.opt.save_period != -1:
            self.wandb.log_model(last.parent, self.opt, epoch, fi,
                best_model=best_fitness == fi)
    if self.clearml:
        if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch
            ) and self.opt.save_period != -1:
            self.clearml.task.update_output_model(model_path=str(last),
                model_name='Latest Model', auto_delete_file=False)
