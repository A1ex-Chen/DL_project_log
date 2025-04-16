def end_epoch(self):
    """
        commit the log_dict, model artifacts and Tables to W&B and flush the log_dict.

        arguments:
        best_result (boolean): Boolean representing if the result of this evaluation is best or not
        """
    if self.wandb_run:
        with all_logging_disabled():
            try:
                wandb.log(self.log_dict)
            except BaseException as e:
                LOGGER.info(
                    f"""An error occurred in wandb logger. The training will proceed without interruption. More info
{e}"""
                    )
                self.wandb_run.finish()
                self.wandb_run = None
            self.log_dict = {}
