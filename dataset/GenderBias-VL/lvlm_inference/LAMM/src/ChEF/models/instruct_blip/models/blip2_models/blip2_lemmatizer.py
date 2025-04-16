@property
def lemmatizer(self):
    if self._lemmatizer is None:
        try:
            import spacy
            self._lemmatizer = spacy.load('en_core_web_sm')
        except ImportError:
            logging.error(
                """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
            exit(1)
    return self._lemmatizer
