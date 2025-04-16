def on_train_end(self, num_batches_in_epoch: int):
    path_to_best_checkpoint = os.path.join(self.path_to_checkpoints_dir,
        f'epoch-{self.best_epoch:06d}', 'checkpoint.pth')
    self.logger.i(f'The best model is {path_to_best_checkpoint}')
    metric_dict = {'hparam/best_epoch': self.best_epoch,
        'hparam/val_mean_ap': self.best_mean_ap}
    if self.test_evaluator is not None:
        global_batch = self.best_epoch * num_batches_in_epoch
        self.logger.i('Start evaluating for test set')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = Checkpoint.load(path_to_best_checkpoint, device)
        evaluation = self.test_evaluator.evaluate(checkpoint.model,
            returns_coco_result=True)
        test_mean_ap = evaluation.metric_ap.mean_value
        test_coco_mean_mean_ap = evaluation.coco_result.mean_mean_ap
        test_coco_mean_standard_ap = evaluation.coco_result.mean_standard_ap
        test_coco_mean_strict_ap = evaluation.coco_result.mean_strict_ap
        self.logger.i(
            f'mean AP = {test_mean_ap:.4f} at epoch {self.best_epoch}')
        self.logger.i('[PyCOCOTools] mean AP@[.5:.95:.05] = {:.4f}'.format(
            test_coco_mean_mean_ap))
        self.logger.i('[PyCOCOTools] mean AP@0.5 = {:.4f}'.format(
            test_coco_mean_standard_ap))
        self.logger.i('[PyCOCOTools] mean AP@0.75 = {:.4f}'.format(
            test_coco_mean_strict_ap))
        self.summary_writer.add_scalar('mean_ap/test', test_mean_ap,
            global_batch)
        self.summary_writer.add_scalar('coco_mean_mean_ap/test',
            test_coco_mean_mean_ap, global_batch)
        self.summary_writer.add_scalar('coco_mean_standard_ap/test',
            test_coco_mean_standard_ap, global_batch)
        self.summary_writer.add_scalar('coco_mean_strict_ap/test',
            test_coco_mean_strict_ap, global_batch)
        with tempfile.TemporaryDirectory() as path_to_temp_dir:
            path_to_plot = os.path.join(path_to_temp_dir, 'metric-ap.png')
            Plotter.plot_pr_curve(self.model.num_classes, self.model.
                class_to_category_dict, mean_ap=test_mean_ap,
                class_to_ap_dict=evaluation.metric_ap.class_to_value_dict,
                class_to_inter_recall_array_dict=evaluation.
                class_to_inter_recall_array_dict,
                class_to_inter_precision_array_dict=evaluation.
                class_to_inter_precision_array_dict, path_to_plot=path_to_plot)
            self.summary_writer.add_image('pr_curve/test', to_tensor(Image.
                open(path_to_plot)), global_batch)
        metric_dict.update({'hparam/test_mean_ap': test_mean_ap,
            'hparam/test_coco_mean_mean_ap': test_coco_mean_mean_ap,
            'hparam/test_coco_mean_standard_ap': test_coco_mean_standard_ap,
            'hparam/test_coco_mean_strict_ap': test_coco_mean_strict_ap})
    logs = self.db.select_log_table()
    detection_logs = self.db.select_detection_log_table()
    global_batches = [log.global_batch for log in logs if log.epoch > 0]
    losses = [log.avg_loss for log in logs if log.epoch > 0]
    anchor_objectness_losses = [log.avg_anchor_objectness_loss for log in
        detection_logs]
    anchor_transformer_losses = [log.avg_anchor_transformer_loss for log in
        detection_logs]
    proposal_class_losses = [log.avg_proposal_class_loss for log in
        detection_logs]
    proposal_transformer_losses = [log.avg_proposal_transformer_loss for
        log in detection_logs]
    legend_to_losses_and_color_dict = {'loss': (losses, 'orange'),
        'anchor objectness loss': (anchor_objectness_losses, 'b--'),
        'anchor transformer loss': (anchor_transformer_losses, 'g--'),
        'proposal class loss': (proposal_class_losses, 'c--'),
        'proposal transformer loss': (proposal_transformer_losses, 'm--')}
    path_to_loss_plot = os.path.join(self.path_to_checkpoints_dir, 'loss.png')
    Plotter.plot_loss_curve(global_batches, legend_to_losses_and_color_dict,
        path_to_loss_plot)
    status = DB.Log.Status.STOPPED if self.is_terminated(
        ) else DB.Log.Status.FINISHED
    self.db.update_log_table_latest_status(status)
    self.summary_writer.add_hparams(hparam_dict=self.config.
        to_hyper_param_dict(), metric_dict=metric_dict)
    self.summary_writer.close()
