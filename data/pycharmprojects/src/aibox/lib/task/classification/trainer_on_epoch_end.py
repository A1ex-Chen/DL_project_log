def on_epoch_end(self, epoch: int, num_batches_in_epoch: int):
    if epoch % self.config.num_epochs_to_validate == 0 or epoch in [self.
        config.step_lr_sizes] or epoch == self.config.num_epochs_to_finish:
        global_batch = epoch * num_batches_in_epoch
        path_to_checkpoint = os.path.join(self.path_to_checkpoints_dir,
            f'epoch-{epoch:06d}', 'checkpoint.pth')
        os.makedirs(os.path.dirname(path_to_checkpoint), exist_ok=True)
        Checkpoint.save(Checkpoint(epoch, self.model, self.optimizer),
            path_to_checkpoint)
        self.logger.i(f'Model has been saved to {path_to_checkpoint}')
        self.logger.i('Start evaluating for validation set')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = Checkpoint.load(path_to_checkpoint, device)
        evaluation = self.val_evaluator.evaluate(checkpoint.model)
        accuracy = evaluation.accuracy
        self.logger.i(f'Accuracy = {accuracy:.4f}')
        self.summary_writer.add_scalar('accuracy/val', accuracy, global_batch)
        avg_loss = sum(self.losses) / len(self.losses)
        categories = ['mean']
        aucs = [evaluation.metric_auc.mean_value]
        sensitivities = [evaluation.metric_sensitivity.mean_value]
        specificities = [evaluation.metric_specificity.mean_value]
        for cls in range(1, self.model.num_classes):
            categories.append(self.model.class_to_category_dict[cls])
            aucs.append(evaluation.metric_auc.class_to_value_dict[cls])
            sensitivities.append(evaluation.metric_sensitivity.
                class_to_value_dict[cls])
            specificities.append(evaluation.metric_specificity.
                class_to_value_dict[cls])
        metrics = DB.Checkpoint.Metrics(DB.Checkpoint.Metrics.Overall(
            accuracy, evaluation.avg_recall, evaluation.avg_precision,
            evaluation.avg_f1_score), DB.Checkpoint.Metrics.Specific(
            categories, aucs, sensitivities, specificities))
        self.db.insert_checkpoint_table(DB.Checkpoint(epoch, avg_loss, metrics)
            )
        if len(self.checkpoint_infos) > self.config.max_num_checkpoints - 1:
            reserved_checkpoint_infos = [checkpoint_info for
                checkpoint_info in self.checkpoint_infos if checkpoint_info
                .epoch in self.config.step_lr_sizes or checkpoint_info.
                epoch == self.config.num_epochs_to_finish]
            reserved_epochs = [checkpoint_info.epoch for checkpoint_info in
                reserved_checkpoint_infos]
            self.checkpoint_infos = Trainer.Callback.CheckpointInfo.sorted([
                checkpoint_info for checkpoint_info in self.
                checkpoint_infos if checkpoint_info.epoch not in
                reserved_epochs])
            removing_checkpoint_info: Trainer.Callback.CheckpointInfo = (self
                .checkpoint_infos.pop())
            removing_epoch = removing_checkpoint_info.epoch
            path_to_removing_epoch_dir = os.path.join(self.
                path_to_checkpoints_dir, f'epoch-{removing_epoch:06d}')
            shutil.rmtree(path_to_removing_epoch_dir)
            self.db.update_checkpoint_table_is_available_for_epoch(is_available
                =False, epoch=removing_checkpoint_info.epoch)
            self.checkpoint_infos.extend(reserved_checkpoint_infos)
        self.checkpoint_infos.append(self.CheckpointInfo(epoch, avg_loss,
            accuracy))
        if accuracy >= self.best_accuracy:
            last_best_epoch = self.best_epoch
            if last_best_epoch is not None:
                self.db.update_checkpoint_table_is_best_for_epoch(is_best=
                    False, epoch=last_best_epoch)
            self.db.update_checkpoint_table_is_best_for_epoch(is_best=True,
                epoch=epoch)
            self.best_accuracy = accuracy
            self.best_epoch = epoch
        self.logger.i(
            f'best Accuracy = {self.best_accuracy:.4f} at epoch {self.best_epoch}'
            )
        path_to_epoch_dir = os.path.join(self.path_to_checkpoints_dir,
            f'epoch-{epoch:06d}')
        path_to_plot_dir = os.path.join(path_to_epoch_dir)
        os.makedirs(path_to_plot_dir, exist_ok=True)
        path_to_plot = os.path.join(path_to_plot_dir, 'metric-auc.png')
        Plotter.plot_roc_curve(self.model.num_classes, self.model.
            class_to_category_dict, macro_average_auc=evaluation.metric_auc
            .mean_value, class_to_auc_dict=evaluation.metric_auc.
            class_to_value_dict, class_to_fpr_array_dict=evaluation.
            class_to_fpr_array_dict, class_to_tpr_array_dict=evaluation.
            class_to_tpr_array_dict, path_to_plot=path_to_plot)
        self.summary_writer.add_image('roc_curve/val', to_tensor(Image.open
            (path_to_plot)), global_batch)
        path_to_plot = os.path.join(path_to_plot_dir, 'confusion-matrix.png')
        Plotter.plot_confusion_matrix(evaluation.confusion_matrix, self.
            model.class_to_category_dict, path_to_plot)
        self.summary_writer.add_image('confusion_matrix/val', to_tensor(
            Image.open(path_to_plot)), global_batch)
