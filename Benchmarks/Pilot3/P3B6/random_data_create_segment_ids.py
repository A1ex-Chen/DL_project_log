def create_segment_ids(self, length, num_docs):
    segment_ids = [self.empty_segment_id(length) for _ in range(num_docs)]
    return torch.stack(segment_ids)
