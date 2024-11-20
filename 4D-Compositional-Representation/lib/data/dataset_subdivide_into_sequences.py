def subdivide_into_sequences(self, models, models_len):
    """ Subdivides model sequence into smaller sequences.

        Args:
            models (list): list of model names
            models_len (list): list of lengths of model sequences
        """
    length_sequence = self.length_sequence
    n_files_per_sequence = self.n_files_per_sequence
    offset_sequence = self.offset_sequence
    models_len = [(l - offset_sequence) for l in models_len]
    if n_files_per_sequence > 0:
        models_len = [min(n_files_per_sequence, l) for l in models_len]
    models_out = []
    start_idx = []
    for idx, model in enumerate(models):
        for n in range(0, models_len[idx] - length_sequence + 1):
            models_out.append(model)
            start_idx.append(n + offset_sequence)
    return models_out, start_idx
