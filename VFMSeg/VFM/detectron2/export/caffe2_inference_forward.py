def forward(self, batched_inputs):
    c2_inputs = self._convert_inputs(batched_inputs)
    c2_results = self.protobuf_model(c2_inputs)
    c2_results = dict(zip(self.protobuf_model.net.Proto().external_output,
        c2_results))
    return self._convert_outputs(batched_inputs, c2_inputs, c2_results)
