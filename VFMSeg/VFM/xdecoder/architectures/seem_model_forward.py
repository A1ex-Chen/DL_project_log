def forward(self, batched_inputs, mode='default'):
    if self.training:
        losses = {}
        if self.task_switch['mask']:
            losses_seg = self.forward_seg(batched_inputs)
            losses.update(losses_seg)
        if self.task_switch['openimage'] and self.task_switch['openimage'][
            'mask']:
            losses_openimage = self.forward_openimage(batched_inputs[
                'openimage'])
            losses_openimage = {key.replace('mask', 'openimage'): value for
                key, value in losses_openimage.items()}
            losses_openimage = {key.replace('grounding',
                'grounding_openimage'): value for key, value in
                losses_openimage.items()}
            losses.update(losses_openimage)
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                losses.pop(k)
        return losses
    elif mode == 'interactive':
        return self.evaluate_interactive(batched_inputs)
    elif mode == 'grounding_spatial':
        return self.evaluate_grounding_sptial(batched_inputs, mode)
    elif mode in ['grounding_phrasecut', 'grounding_refcoco']:
        return self.evaluate_grounding(batched_inputs, mode)
    else:
        return self.evaluate(batched_inputs)
