def clear_conditioned_layers(self) ->None:
    for layer in self._get_decoder_layers():
        layer.condition_vis_x(None)
        layer.condition_media_locations(None)
        layer.condition_attend_previous(None)
