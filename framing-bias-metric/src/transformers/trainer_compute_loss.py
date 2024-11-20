def compute_loss(self, model, inputs):
    """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
    outputs = model(**inputs)
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]
    return outputs['loss'] if isinstance(outputs, dict) else outputs[0]
