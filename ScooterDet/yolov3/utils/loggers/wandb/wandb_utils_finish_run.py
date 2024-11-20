def finish_run(self):
    """Log metrics if any and finish the current W&B run."""
    if self.wandb_run:
        if self.log_dict:
            with all_logging_disabled():
                wandb.log(self.log_dict)
        wandb.run.finish()
        LOGGER.warning(DEPRECATION_WARNING)
