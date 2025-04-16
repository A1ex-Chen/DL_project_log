@property
def remote_dataset(self):
    data_dict = None
    if self.clearml:
        data_dict = self.clearml.data_dict
    if self.wandb:
        data_dict = self.wandb.data_dict
    if self.comet_logger:
        data_dict = self.comet_logger.data_dict
    return data_dict
