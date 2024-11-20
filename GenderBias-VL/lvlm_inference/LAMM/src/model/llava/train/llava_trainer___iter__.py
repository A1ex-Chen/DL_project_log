def __iter__(self):
    if self.group_by_modality:
        indices = get_modality_length_grouped_indices(self.lengths, self.
            batch_size, self.world_size, generator=self.generator)
    else:
        indices = get_length_grouped_indices(self.lengths, self.batch_size,
            self.world_size, generator=self.generator)
    return iter(indices)
