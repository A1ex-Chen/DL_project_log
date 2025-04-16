def get_output(self, device) ->Dict[str, torch.tensor]:
    output = self._feature_outputs[device]
    self._feature_outputs[device] = OrderedDict()
    return output
