def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.
    Tensor, Any]]) ->torch.Tensor:
    """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
    model.train()
    inputs = self._prepare_inputs(inputs)
    if self.args.fp16 and _use_native_amp:
        with autocast():
            loss = self.compute_loss(model, inputs)
    else:
        loss = self.compute_loss(model, inputs)
    if self.args.n_gpu > 1:
        loss = loss.mean()
    if self.args.gradient_accumulation_steps > 1:
        loss = loss / self.args.gradient_accumulation_steps
    if self.args.fp16 and _use_native_amp:
        self.scaler.scale(loss).backward()
    elif self.args.fp16 and _use_apex:
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    return loss.detach()
