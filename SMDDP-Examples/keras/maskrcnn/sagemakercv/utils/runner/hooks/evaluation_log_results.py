def log_results(self, stat_dict, logger):
    for iou, iou_dict in stat_dict.items():
        for stat, value in iou_dict.items():
            logger.info(f'{iou} {stat}: {value}')
