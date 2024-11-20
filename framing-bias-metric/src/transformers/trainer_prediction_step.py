def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.
    Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[List[
    str]]=None) ->Tuple[Optional[float], Optional[torch.Tensor], Optional[
    torch.Tensor]]:
    """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
    has_labels = all(inputs.get(k) is not None for k in self.label_names)
    inputs = self._prepare_inputs(inputs)
    if ignore_keys is None:
        if hasattr(self.model, 'config'):
            ignore_keys = getattr(self.model.config,
                'keys_to_ignore_at_inference', [])
        else:
            ignore_keys = []
    with torch.no_grad():
        if self.args.fp16 and _use_native_amp:
            with autocast():
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
        if has_labels:
            if isinstance(outputs, dict):
                loss = outputs['loss'].mean().detach()
                logits = tuple(v for k, v in outputs.items() if k not in 
                    ignore_keys + ['loss'])
            else:
                loss = outputs[0].mean().detach()
                logits = outputs[1:]
        else:
            loss = None
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in
                    ignore_keys)
            else:
                logits = outputs
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index if has_labels else 
                self.args.past_index - 1]
    if prediction_loss_only:
        return loss, None, None
    logits = nested_detach(logits)
    if len(logits) == 1:
        logits = logits[0]
    if has_labels:
        labels = nested_detach(tuple(inputs.get(name) for name in self.
            label_names))
        if len(labels) == 1:
            labels = labels[0]
    else:
        labels = None
    return loss, logits, labels
