def on_fit_epoch_end(trainer):
    """Reports model information to logger at the end of an epoch."""
    if (task := Task.current_task()):
        task.get_logger().report_scalar(title='Epoch Time', series=
            'Epoch Time', value=trainer.epoch_time, iteration=trainer.epoch)
        for k, v in trainer.metrics.items():
            task.get_logger().report_scalar('val', k, v, iteration=trainer.
                epoch)
        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers
            for k, v in model_info_for_loggers(trainer).items():
                task.get_logger().report_single_value(k, v)
