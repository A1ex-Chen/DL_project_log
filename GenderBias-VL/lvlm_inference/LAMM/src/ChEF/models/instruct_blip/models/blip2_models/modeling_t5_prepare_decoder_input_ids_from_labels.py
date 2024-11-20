def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
    return self._shift_right(labels)
