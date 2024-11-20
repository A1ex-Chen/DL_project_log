def __setstate__(self, d: Dict) ->None:
    self.__dict__ = d
    self.spm_source, self.spm_target = (load_spm(f) for f in self.spm_files)
    self.current_spm = self.spm_source
    self._setup_normalizer()
