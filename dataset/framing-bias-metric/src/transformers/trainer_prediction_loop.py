def prediction_loop(self, dataloader: DataLoader, description: str,
    prediction_loss_only: Optional[bool]=None, ignore_keys: Optional[List[
    str]]=None) ->PredictionOutput:
    """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
    if not isinstance(dataloader.dataset, collections.abc.Sized):
        raise ValueError('dataset must implement __len__')
    prediction_loss_only = (prediction_loss_only if prediction_loss_only is not
        None else self.args.prediction_loss_only)
    model = self.model
    if self.args.n_gpu > 1 and not self.args.model_parallel:
        model = torch.nn.DataParallel(model)
    batch_size = dataloader.batch_size
    num_examples = self.num_examples(dataloader)
    logger.info('***** Running %s *****', description)
    logger.info('  Num examples = %d', num_examples)
    logger.info('  Batch size = %d', batch_size)
    losses_host: torch.Tensor = None
    preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
    labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
    world_size = 1
    if is_torch_tpu_available():
        world_size = xm.xrt_world_size()
    elif self.args.local_rank != -1:
        world_size = torch.distributed.get_world_size()
    world_size = max(1, world_size)
    eval_losses_gatherer = DistributedTensorGatherer(world_size,
        num_examples, make_multiple_of=batch_size)
    if not prediction_loss_only:
        preds_gatherer = DistributedTensorGatherer(world_size, num_examples)
        labels_gatherer = DistributedTensorGatherer(world_size, num_examples)
    model.eval()
    if is_torch_tpu_available():
        dataloader = pl.ParallelLoader(dataloader, [self.args.device]
            ).per_device_loader(self.args.device)
    if self.args.past_index >= 0:
        self._past = None
    self.callback_handler.eval_dataloader = dataloader
    for step, inputs in enumerate(dataloader):
        loss, logits, labels = self.prediction_step(model, inputs,
            prediction_loss_only, ignore_keys=ignore_keys)
        if loss is not None:
            losses = loss.repeat(batch_size)
            losses_host = losses if losses_host is None else torch.cat((
                losses_host, losses), dim=0)
        if logits is not None:
            preds_host = logits if preds_host is None else nested_concat(
                preds_host, logits, padding_index=-100)
        if labels is not None:
            labels_host = labels if labels_host is None else nested_concat(
                labels_host, labels, padding_index=-100)
        self.control = self.callback_handler.on_prediction_step(self.args,
            self.state, self.control)
        if self.args.eval_accumulation_steps is not None and (step + 1
            ) % self.args.eval_accumulation_steps == 0:
            eval_losses_gatherer.add_arrays(self._gather_and_numpify(
                losses_host, 'eval_losses'))
            if not prediction_loss_only:
                preds_gatherer.add_arrays(self._gather_and_numpify(
                    preds_host, 'eval_preds'))
                labels_gatherer.add_arrays(self._gather_and_numpify(
                    labels_host, 'eval_label_ids'))
            losses_host, preds_host, labels_host = None, None, None
    if self.args.past_index and hasattr(self, '_past'):
        delattr(self, '_past')
    eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host,
        'eval_losses'))
    if not prediction_loss_only:
        preds_gatherer.add_arrays(self._gather_and_numpify(preds_host,
            'eval_preds'))
        labels_gatherer.add_arrays(self._gather_and_numpify(labels_host,
            'eval_label_ids'))
    eval_loss = eval_losses_gatherer.finalize()
    preds = preds_gatherer.finalize() if not prediction_loss_only else None
    label_ids = labels_gatherer.finalize(
        ) if not prediction_loss_only else None
    if (self.compute_metrics is not None and preds is not None and 
        label_ids is not None):
        metrics = self.compute_metrics(EvalPrediction(predictions=preds,
            label_ids=label_ids))
    else:
        metrics = {}
    if eval_loss is not None:
        metrics['eval_loss'] = eval_loss.mean().item()
    for key in list(metrics.keys()):
        if not key.startswith('eval_'):
            metrics[f'eval_{key}'] = metrics.pop(key)
    return PredictionOutput(predictions=preds, label_ids=label_ids, metrics
        =metrics)
