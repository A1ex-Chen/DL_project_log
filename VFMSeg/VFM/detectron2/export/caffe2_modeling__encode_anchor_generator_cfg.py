def _encode_anchor_generator_cfg(self, predict_net):
    serialized_anchor_generator = io.BytesIO()
    torch.save(self._wrapped_model.anchor_generator,
        serialized_anchor_generator)
    bytes = serialized_anchor_generator.getvalue()
    check_set_pb_arg(predict_net, 'serialized_anchor_generator', 's', bytes)
