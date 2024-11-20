@staticmethod
def check_thres(conf_thres, iou_thres, task):
    """Check whether confidence and iou threshold are best for task val/speed"""
    if task != 'train':
        if task == 'val' or task == 'test':
            if conf_thres > 0.03:
                LOGGER.warning(
                    f'The best conf_thresh when evaluate the model is less than 0.03, while you set it to: {conf_thres}'
                    )
            if iou_thres != 0.65:
                LOGGER.warning(
                    f'The best iou_thresh when evaluate the model is 0.65, while you set it to: {iou_thres}'
                    )
        if task == 'speed' and conf_thres < 0.4:
            LOGGER.warning(
                f'The best conf_thresh when test the speed of the model is larger than 0.4, while you set it to: {conf_thres}'
                )
