def test(self):
    print('Test best model with test set!')
    best_model_path = os.path.join(self.export_root, 'models',
        'best_acc_model.pth'
        ) if self.args.mode == 'train' else self.args.test_model_path
    best_model = torch.load(best_model_path).get('model_state_dict')
    self.model.load_state_dict(best_model)
    self.model.eval()
    average_meter_set = AverageMeterSet()
    with torch.no_grad():
        tqdm_dataloader = tqdm(self.test_loader)
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
        average_metrics = average_meter_set.averages()
        with open(os.path.join(self.export_root, 'logs',
            'test_metrics.json'), 'w') as f:
            json.dump(average_metrics, f, indent=4)
        print(average_metrics)
