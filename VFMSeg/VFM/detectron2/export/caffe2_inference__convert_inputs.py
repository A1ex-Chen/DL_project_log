def _convert_inputs(self, batched_inputs):
    return convert_batched_inputs_to_c2_format(batched_inputs, self.
        size_divisibility, self.device)
