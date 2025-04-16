def on_val_end(self, nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix
    ):
    if self.comet_log_per_class_metrics and self.num_classes > 1:
        for i, c in enumerate(ap_class):
            class_name = self.class_names[c]
            self.experiment.log_metrics({'mAP@.5': ap50[i], 'mAP@.5:.95':
                ap[i], 'precision': p[i], 'recall': r[i], 'f1': f1[i],
                'true_positives': tp[i], 'false_positives': fp[i],
                'support': nt[c]}, prefix=class_name)
    if self.comet_log_confusion_matrix:
        epoch = self.experiment.curr_epoch
        class_names = list(self.class_names.values())
        class_names.append('background')
        num_classes = len(class_names)
        self.experiment.log_confusion_matrix(matrix=confusion_matrix.matrix,
            max_categories=num_classes, labels=class_names, epoch=epoch,
            column_label='Actual Category', row_label='Predicted Category',
            file_name=f'confusion-matrix-epoch-{epoch}.json')
