def validate(self, epoch, accum_iter):
    self.model.eval()
    average_meter_set = AverageMeterSet()
    with torch.no_grad():
        tqdm_dataloader = tqdm(self.val_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]
            metrics = self.calculate_metrics(batch)
            for k, v in metrics.items():
                average_meter_set.update(k, v)
            description_metrics = [('NDCG@%d' % k) for k in self.metric_ks[:3]
                ] + [('Recall@%d' % k) for k in self.metric_ks[:3]]
            description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in
                description_metrics)
            description = description.replace('NDCG', 'N').replace('Recall',
                'R')
            description = description.format(*(average_meter_set[k].avg for
                k in description_metrics))
            tqdm_dataloader.set_description(description)
        log_data = {'state_dict': self._create_state_dict(), 'epoch': epoch +
            1, 'accum_iter': accum_iter}
        log_data.update(average_meter_set.averages())
        self.log_extra_val_info(log_data)
        self.logger_service.log_val(log_data)
