def run(self):
    from transformers import AutoModel, AutoTokenizer
    AutoModel.from_pretrained(self._model, cache_dir=self._cache,
        force_download=self._force)
    AutoTokenizer.from_pretrained(self._model, cache_dir=self._cache,
        force_download=self._force)
