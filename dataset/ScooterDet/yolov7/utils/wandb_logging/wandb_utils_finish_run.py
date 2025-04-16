def finish_run(self):
    if self.wandb_run:
        if self.log_dict:
            wandb.log(self.log_dict)
        wandb.run.finish()
