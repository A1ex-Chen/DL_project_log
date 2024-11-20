def collater(self, samples):
    all_keys = set()
    for s in samples:
        all_keys.update(s)
    shared_keys = all_keys
    for s in samples:
        shared_keys = shared_keys & set(s.keys())
    samples_shared_keys = []
    for s in samples:
        samples_shared_keys.append({k: s[k] for k in s.keys() if k in
            shared_keys})
    return self.datasets[0].collater(samples_shared_keys)
