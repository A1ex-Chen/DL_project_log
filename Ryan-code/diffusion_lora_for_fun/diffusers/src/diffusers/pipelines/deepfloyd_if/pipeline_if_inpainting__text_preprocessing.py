def _text_preprocessing(self, text, clean_caption=False):
    if clean_caption and not is_bs4_available():
        logger.warning(BACKENDS_MAPPING['bs4'][-1].format(
            'Setting `clean_caption=True`'))
        logger.warning('Setting `clean_caption` to False...')
        clean_caption = False
    if clean_caption and not is_ftfy_available():
        logger.warning(BACKENDS_MAPPING['ftfy'][-1].format(
            'Setting `clean_caption=True`'))
        logger.warning('Setting `clean_caption` to False...')
        clean_caption = False
    if not isinstance(text, (tuple, list)):
        text = [text]

    def process(text: str):
        if clean_caption:
            text = self._clean_caption(text)
            text = self._clean_caption(text)
        else:
            text = text.lower().strip()
        return text
    return [process(t) for t in text]
