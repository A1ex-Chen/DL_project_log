def write_logs(self, logs):
    with open(self._path, 'a') as f:
        w = csv.DictWriter(f, self._keys)
        w.writerow(logs)
