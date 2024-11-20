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

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
    inputs = self._prepare_inputs(inputs)
    gen_kwargs = {'max_length': self.data_args.val_max_target_length if 
        self.data_args is not None else self.config.max_length, 'num_beams':
        self.data_args.eval_beams if self.data_args is not None else self.
        config.num_beams}
    if self.args.predict_with_generate and not self.args.prediction_loss_only:
        generated_tokens = self.model.generate(inputs['input_ids'],
            attention_mask=inputs['attention_mask'], **gen_kwargs)
        if generated_tokens.shape[-1] < gen_kwargs['max_length']:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens,
                gen_kwargs['max_length'])
    labels = inputs.pop('labels')
    with torch.no_grad():
        loss, logits = self._compute_loss(model, inputs, labels)
    loss = loss.mean().detach()
    if self.args.prediction_loss_only:
        return loss, None, None
    logits = generated_tokens if self.args.predict_with_generate else logits
    if labels.shape[-1] < gen_kwargs['max_length']:
        labels = self._pad_tensors_to_max_len(labels, gen_kwargs['max_length'])
    return loss, logits, labels
