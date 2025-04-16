def __iter__(self):
    for idx in self.sampler:
        group_id = self.group_ids[idx]
        group_buffer = self.buffer_per_group[group_id]
        group_buffer.append(idx)
        if len(group_buffer) == self.batch_size:
            yield group_buffer[:]
            del group_buffer[:]
