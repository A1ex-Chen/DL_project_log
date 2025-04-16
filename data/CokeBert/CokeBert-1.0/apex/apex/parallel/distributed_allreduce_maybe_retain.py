def allreduce_maybe_retain(self, bucket, bucket_idx=-1):
    allreduced = self.allreduce_bucket(bucket)
    if self.retain_allreduce_buffers:
        if self.allreduce_buffers[bucket_idx] is not None:
            raise RuntimeError(
                'The backward pass is attempting to replace an already-filled allreduce buffer.  This is almost certainly an error.'
                )
        self.allreduce_buffers[bucket_idx] = allreduced
    elif multi_tensor_applier.available:
        multi_tensor_applier(self.multi_tensor_scale, self._overflow_buf, [
            unflatten(allreduced, bucket), bucket], 1.0)
    else:
        for buf, synced in zip(bucket, unflatten(allreduced, bucket)):
            buf.copy_(synced)
