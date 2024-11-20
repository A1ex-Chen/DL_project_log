def validation_epoch_end(self, outputs, prefix='val') ->Dict:
    self.step_count += 1
    losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.
        loss_names}
    loss = losses['loss']
    self.log('validation_loss', loss, on_epoch=True)
    generative_metrics = {k: np.array([x[k] for x in outputs]).mean() for k in
        self.metric_names + ['gen_time', 'gen_len']}
    metric_val = generative_metrics[self.val_metric
        ] if self.val_metric in generative_metrics else losses[self.val_metric]
    metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
    generative_metrics.update({k: v.item() for k, v in losses.items()})
    losses.update(generative_metrics)
    all_metrics = {f'{prefix}_avg_{k}': x for k, x in losses.items()}
    all_metrics['step_count'] = self.step_count
    self.metrics[prefix].append(all_metrics)
    preds = flatten_list([x['preds'] for x in outputs])
    print(self.val_metric)
    return {'log': all_metrics, 'preds': preds, f'{prefix}_loss': loss,
        f'{prefix}_{self.val_metric}': metric_tensor}
