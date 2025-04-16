@staticmethod
def write_metrics(loss_dict: Mapping[str, torch.Tensor], data_time: float,
    prefix: str='') ->None:
    """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
    metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
    metrics_dict['data_time'] = data_time
    all_metrics_dict = comm.gather(metrics_dict)
    if comm.is_main_process():
        storage = get_event_storage()
        data_time = np.max([x.pop('data_time') for x in all_metrics_dict])
        storage.put_scalar('data_time', data_time)
        metrics_dict = {k: np.mean([x.get(k, 0.0) for x in all_metrics_dict
            ]) for k in all_metrics_dict[0].keys()}
        total_losses_reduced = sum(metrics_dict.values())
        if not np.isfinite(total_losses_reduced):
            raise FloatingPointError(
                f"""Loss became infinite or NaN at iteration={storage.iter}!
loss_dict = {metrics_dict}"""
                )
        storage.put_scalar('{}total_loss'.format(prefix), total_losses_reduced)
        if len(metrics_dict) > 1:
            storage.put_scalars(**metrics_dict)
