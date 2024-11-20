def sort_by_length(self):
    """
        Sorts dataset by the sequence length.
        """
    self.lengths, indices = self.lengths.sort(descending=True)
    self.src = [self.src[idx] for idx in indices]
    self.tgt = [self.tgt[idx] for idx in indices]
    self.src_lengths = [self.src_lengths[idx] for idx in indices]
    self.tgt_lengths = [self.tgt_lengths[idx] for idx in indices]
    self.indices = indices.tolist()
    self.sorted = True
