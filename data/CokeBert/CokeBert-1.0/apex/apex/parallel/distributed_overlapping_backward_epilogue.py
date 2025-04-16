def overlapping_backward_epilogue():
    self.reduction_stream.record_event(self.reduction_event)
    torch.cuda.current_stream().wait_event(self.reduction_event)
    if self.next_bucket != self.num_buckets:
        raise RuntimeError(
            'In epilogue, next_bucket ({}) != num_buckets ({}).  '.format(
            self.next_bucket, self.num_buckets),
            'This probably indicates some buckets were not allreduced.')
    for actual, expected in zip(self.buckets_ready_size, self.bucket_sizes):
        if actual != expected:
            raise RuntimeError('Some param buckets were not allreduced.')
