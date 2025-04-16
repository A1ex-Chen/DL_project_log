def _setup_normalizer(self):
    try:
        from sacremoses import MosesPunctNormalizer
        self.punc_normalizer = MosesPunctNormalizer(self.source_lang).normalize
    except (ImportError, FileNotFoundError):
        warnings.warn('Recommended: pip install sacremoses.')
        self.punc_normalizer = lambda x: x
