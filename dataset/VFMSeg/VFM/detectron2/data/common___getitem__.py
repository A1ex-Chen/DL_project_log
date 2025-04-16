def __getitem__(self, idx):
    if self._serialize:
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr])
        return pickle.loads(bytes)
    elif self._copy:
        return copy.deepcopy(self._lst[idx])
    else:
        return self._lst[idx]
