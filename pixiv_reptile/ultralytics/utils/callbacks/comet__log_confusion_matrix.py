def _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch):
    """Log the confusion matrix to Comet experiment."""
    conf_mat = trainer.validator.confusion_matrix.matrix
    names = list(trainer.data['names'].values()) + ['background']
    experiment.log_confusion_matrix(matrix=conf_mat, labels=names,
        max_categories=len(names), epoch=curr_epoch, step=curr_step)
