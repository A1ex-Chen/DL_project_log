def _write_metrics(self, loss_dict: Mapping[str, torch.Tensor], data_time:
    float, prefix: str='') ->None:
    SimpleTrainer.write_metrics(loss_dict, data_time, prefix)
