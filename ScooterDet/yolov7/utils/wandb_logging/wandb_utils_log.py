def log(self, log_dict):
    if self.wandb_run:
        for key, value in log_dict.items():
            self.log_dict[key] = value
