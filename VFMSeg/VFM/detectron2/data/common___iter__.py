def __iter__(self):
    for d in self.dataset:
        w, h = d['width'], d['height']
        bucket_id = 0 if w > h else 1
        bucket = self._buckets[bucket_id]
        bucket.append(d)
        if len(bucket) == self.batch_size:
            data = bucket[:]
            del bucket[:]
            yield data
