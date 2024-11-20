def dump(self, file_path: str):
    self._C.dump(stream=open(file_path, 'w'))
