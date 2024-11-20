def comm_ready_buckets(self, param):
    bucket_idx, bucket_loc = self.param_id_to_bucket[id(param)]
    if self.buckets[bucket_idx][bucket_loc] is not None:
        raise RuntimeError(
            'The backward pass is attempting to replace an already-filled bucket slot.  This is almost certainly an error.'
            )
    self.buckets[bucket_idx][bucket_loc] = param.grad.data
    self.buckets_ready_size[bucket_idx] += 1
    if self.buckets_ready_size[bucket_idx] == self.bucket_sizes[bucket_idx]:
        if bucket_idx == self.next_bucket:
            torch.cuda.current_stream().record_event(self.reduction_event)
            self.reduction_stream.wait_event(self.reduction_event)
            with torch.cuda.stream(self.reduction_stream):
                self.allreduce_maybe_retain(self.buckets[bucket_idx],
                    bucket_idx)
                self.next_bucket += 1
                if len(self.ready_buckets_not_reduced) > 0:
                    sorted_todo = sorted(self.ready_buckets_not_reduced)
                    for i in sorted_todo:
                        if i > self.next_bucket:
                            break
                        elif i == self.next_bucket:
                            self.allreduce_maybe_retain(self.buckets[i], i)
                            self.ready_buckets_not_reduced.remove(i)
                            self.next_bucket += 1
                        else:
                            raise ValueError(
                                'i should always be >= next_bucket')
        else:
            self.ready_buckets_not_reduced.add(bucket_idx)
