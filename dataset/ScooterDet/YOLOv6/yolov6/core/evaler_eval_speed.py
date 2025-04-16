def eval_speed(self, task):
    """Evaluate model inference speed."""
    if task != 'train':
        n_samples = self.speed_result[0].item()
        pre_time, inf_time, nms_time = 1000 * self.speed_result[1:].cpu(
            ).numpy() / n_samples
        for n, v in zip(['pre-process', 'inference', 'NMS'], [pre_time,
            inf_time, nms_time]):
            LOGGER.info('Average {} time: {:.2f} ms'.format(n, v))
