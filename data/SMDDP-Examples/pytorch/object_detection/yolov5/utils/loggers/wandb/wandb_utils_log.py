def log(self, log_dict):
    """
        save the metrics to the logging dictionary

        arguments:
        log_dict (Dict) -- metrics/media to be logged in current step
        """
    if self.wandb_run:
        for key, value in log_dict.items():
            self.log_dict[key] = value
