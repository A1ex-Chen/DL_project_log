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
        mean_ap = evaluation.metric_ap.mean_value
        self.logger.i(f'mean AP = {mean_ap:.4f}')
        self.summary_writer.add_scalar('mean_ap/val', mean_ap, global_batch)
        avg_loss = sum(self.losses) / len(self.losses)
        categories = ['mean']
        aps = [evaluation.metric_ap.mean_value]
        f1_scores = [evaluation.metric_top_f1_score.mean_value]
        precisions = [evaluation.metric_precision_at_top_f1_score.mean_value]
        recalls = [evaluation.metric_recall_at_top_f1_score.mean_value]
        accuracies = [evaluation.metric_accuracy_at_top_f1_score.mean_value]
        for cls in range(1, self.model.num_classes):
            categories.append(self.model.class_to_category_dict[cls])
            aps.append(evaluation.metric_ap.class_to_value_dict[cls])
            f1_scores.append(evaluation.metric_top_f1_score.
                class_to_value_dict[cls])
            precisions.append(evaluation.metric_precision_at_top_f1_score.
                class_to_value_dict[cls])
            recalls.append(evaluation.metric_recall_at_top_f1_score.
                class_to_value_dict[cls])
            accuracies.append(evaluation.metric_accuracy_at_top_f1_score.
                class_to_value_dict[cls])
        metrics = DB.Checkpoint.Metrics(DB.Checkpoint.Metrics.Overall(), DB
            .Checkpoint.Metrics.Specific(categories, aps, f1_scores,
            precisions, recalls, accuracies))
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
            mean_ap))
        if mean_ap >= self.best_mean_ap:
            last_best_epoch = self.best_epoch
            if last_best_epoch is not None:
                self.db.update_checkpoint_table_is_best_for_epoch(is_best=
                    False, epoch=last_best_epoch)
            self.db.update_checkpoint_table_is_best_for_epoch(is_best=True,
                epoch=epoch)
            self.best_mean_ap = mean_ap
            self.best_epoch = epoch
        self.logger.i(
            f'best mean AP = {self.best_mean_ap:.4f} at epoch {self.best_epoch}'
            )
        path_to_epoch_dir = os.path.join(self.path_to_checkpoints_dir,
            f'epoch-{epoch:06d}')
        path_to_plot_dir = os.path.join(path_to_epoch_dir, 'quality-{:s}'.
            format(evaluation.quality.value), 'size-{:s}'.format(evaluation
            .size.value))
        os.makedirs(path_to_plot_dir)
        path_to_plot = os.path.join(path_to_plot_dir, 'metric-ap.png')
        Plotter.plot_pr_curve(self.model.num_classes, self.model.
            class_to_category_dict, mean_ap=evaluation.metric_ap.mean_value,
            class_to_ap_dict=evaluation.metric_ap.class_to_value_dict,
            class_to_inter_recall_array_dict=evaluation.
            class_to_inter_recall_array_dict,
            class_to_inter_precision_array_dict=evaluation.
            class_to_inter_precision_array_dict, path_to_plot=path_to_plot)
        self.summary_writer.add_image('pr_curve/val', to_tensor(Image.open(
            path_to_plot)), global_batch)
        for target_class in range(1, self.model.num_classes):
            Plotter.plot_thresh_vs_pr_bar(self.model.num_classes, self.
                model.class_to_category_dict, class_to_ap_dict=evaluation.
                metric_ap.class_to_value_dict, class_to_recall_array_dict=
                evaluation.class_to_recall_array_dict,
                class_to_precision_array_dict=evaluation.
                class_to_precision_array_dict, class_to_f1_score_array_dict
                =evaluation.class_to_f1_score_array_dict,
                class_to_prob_array_dict=evaluation.
                class_to_prob_array_dict, path_to_placeholder_to_plot=os.
                path.join(path_to_plot_dir, 'thresh-{}.png'))
