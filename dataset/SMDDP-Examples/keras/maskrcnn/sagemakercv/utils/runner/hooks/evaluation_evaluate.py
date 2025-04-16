@master_only
def evaluate(self, imgIds_mpi_list, box_predictions_mpi_list,
    mask_predictions_mpi_list, logger, iteration, tensorboard_dir=None):
    imgIds = []
    box_predictions = []
    mask_predictions = []
    for i in imgIds_mpi_list:
        imgIds.extend(i)
    for i in box_predictions_mpi_list:
        box_predictions.extend(i)
    predictions = {'bbox': box_predictions}
    if self.include_mask_head:
        for i in mask_predictions_mpi_list:
            mask_predictions.extend(i)
        predictions['segm'] = mask_predictions
    logger.info('Running Evaluation for {} images'.format(len(set(imgIds))))
    stat_dict = evaluation.evaluate_coco_predictions(self.annotations_file,
        predictions.keys(), predictions, self.verbose)
    logger.info(f'{stat_dict}')
    self.log_results(stat_dict, logger)
    if self.tensorboard:
        self.tensorboard_writer(stat_dict, iteration, tensorboard_dir)
