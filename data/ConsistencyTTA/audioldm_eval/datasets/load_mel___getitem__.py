def __getitem__(self, index):
    filename = self.datalist[index]
    waveform = self.read_from_file(filename)
    if waveform.shape[-1] < 1:
        raise ValueError('empty file %s' % filename)
    return waveform, os.path.basename(filename)
