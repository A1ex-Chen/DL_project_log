def on_train_end(self, num_batches_in_epoch: int):
    path_to_best_checkpoint = os.path.join(self.path_to_checkpoints_dir,
        f'epoch-{self.best_epoch:06d}', 'checkpoint.pth')
    self.logger.i(f'The best model is {path_to_best_checkpoint}')
    metric_dict = {'hparam/best_epoch': self.best_epoch,
        'hparam/val_accuracy': self.best_accuracy}
    if self.test_evaluator is not None:
        global_batch = self.best_epoch * num_batches_in_epoch
        self.logger.i('Start evaluating for test set')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = Checkpoint.load(path_to_best_checkpoint, device)
        evaluation = self.test_evaluator.evaluate(checkpoint.model)
        test_accuracy = evaluation.accuracy
        self.logger.i(
            f'Accuracy = {test_accuracy:.4f} at epoch {self.best_epoch}')
        self.summary_writer.add_scalar('accuracy/test', test_accuracy,
            global_batch)
        with tempfile.TemporaryDirectory() as path_to_temp_dir:
            path_to_plot = os.path.join(path_to_temp_dir, 'metric-auc.png')
            Plotter.plot_roc_curve(self.model.num_classes, self.model.
                class_to_category_dict, macro_average_auc=evaluation.
                metric_auc.mean_value, class_to_auc_dict=evaluation.
                metric_auc.class_to_value_dict, class_to_fpr_array_dict=
                evaluation.class_to_fpr_array_dict, class_to_tpr_array_dict
                =evaluation.class_to_tpr_array_dict, path_to_plot=path_to_plot)
            self.summary_writer.add_image('roc_curve/test', to_tensor(Image
                .open(path_to_plot)), global_batch)
            path_to_plot = os.path.join(path_to_temp_dir,
                'confusion-matrix.png')
            Plotter.plot_confusion_matrix(evaluation.confusion_matrix, self
                .model.class_to_category_dict, path_to_plot)
            self.summary_writer.add_image('confusion_matrix/test',
                to_tensor(Image.open(path_to_plot)), global_batch)
        metric_dict.update({'hparam/test_accuracy': test_accuracy})
    logs = self.db.select_log_table()
    global_batches = [log.global_batch for log in logs if log.epoch > 0]
    losses = [log.avg_loss for log in logs if log.epoch > 0]
    legend_to_losses_and_color_dict = {'loss': (losses, 'orange')}
    path_to_loss_plot = os.path.join(self.path_to_checkpoints_dir, 'loss.png')
    Plotter.plot_loss_curve(global_batches, legend_to_losses_and_color_dict,
        path_to_loss_plot)
    status = DB.Log.Status.STOPPED if self.is_terminated(
        ) else DB.Log.Status.FINISHED
    self.db.update_log_table_latest_status(status)
    self.summary_writer.add_hparams(hparam_dict=self.config.
        to_hyper_param_dict(), metric_dict=metric_dict)
    self.summary_writer.close()
