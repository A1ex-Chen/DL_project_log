def _tokenize(self, text, lang='en', bypass_tokenizer=False):
    """
        Tokenize a string given language code. For Chinese, Japanese and Thai, we use a language specific tokenizerself. Otherwise, we use Moses.

        Details of tokenization:
        - [sacremoses](https://github.com/alvations/sacremoses): port of Moses
            - Install with `pip install sacremoses`
        - [pythainlp](https://github.com/PyThaiNLP/pythainlp): Thai tokenizer
            - Install with `pip install pythainlp`
        - [kytea](https://github.com/chezou/Mykytea-python): Japanese tokenizer, wrapper of [KyTea](https://github.com/neubig/kytea)
            - Install with the following steps:
            ```
            git clone git@github.com:neubig/kytea.git && cd kytea
            autoreconf -i
            ./configure --prefix=$HOME/local
            make && make install
            pip install kytea
            ```
        - [jieba](https://github.com/fxsjy/jieba): Chinese tokenizer *
            - Install with `pip install jieba`

        \\* The original XLM used [Stanford Segmenter](https://nlp.stanford.edu/software/stanford-segmenter-2018-10-16.zip).
        However, the wrapper (`nltk.tokenize.stanford_segmenter`) is slow due to JVM overhead, and it will be deprecated.
        Jieba is a lot faster and pip-installable. Note there is some mismatch with the Stanford Segmenter. It should be fine
        if you fine-tune the model with Chinese supervisionself. If you want the same exact behaviour, use the original XLM
        [preprocessing script](https://github.com/facebookresearch/XLM/tree/master/tools) to tokenize the sentence externally,
        and set `bypass_tokenizer=True` to bypass the tokenizer.

        Args:
            - lang: ISO language code (default = 'en') (string). Languages should belong of the model supported languages. However, we don't enforce it.
            - bypass_tokenizer: Allow users to preprocess and tokenize the sentences externally (default = False)  (bool). If True, we only apply BPE.

        Returns:
            List of tokens.
        """
    if lang and self.lang2id and lang not in self.lang2id:
        logger.error(
            'Supplied language code not found in lang2id mapping. Please check that your language is supported by the loaded pretrained model.'
            )
    if bypass_tokenizer:
        text = text.split()
    elif lang not in self.lang_with_custom_tokenizer:
        text = self.moses_pipeline(text, lang=lang)
        if lang == 'ro':
            text = romanian_preprocessing(text)
        text = self.moses_tokenize(text, lang=lang)
    elif lang == 'th':
        text = self.moses_pipeline(text, lang=lang)
        try:
            if 'pythainlp' not in sys.modules:
                from pythainlp.tokenize import word_tokenize as th_word_tokenize
            else:
                th_word_tokenize = sys.modules['pythainlp'].word_tokenize
        except (AttributeError, ImportError) as e:
            logger.error(
                'Make sure you install PyThaiNLP (https://github.com/PyThaiNLP/pythainlp) with the following steps'
                )
            logger.error('1. pip install pythainlp')
            raise e
        text = th_word_tokenize(text)
    elif lang == 'zh':
        try:
            if 'jieba' not in sys.modules:
                import jieba
            else:
                jieba = sys.modules['jieba']
        except (AttributeError, ImportError) as e:
            logger.error(
                'Make sure you install Jieba (https://github.com/fxsjy/jieba) with the following steps'
                )
            logger.error('1. pip install jieba')
            raise e
        text = ' '.join(jieba.cut(text))
        text = self.moses_pipeline(text, lang=lang)
        text = text.split()
    elif lang == 'ja':
        text = self.moses_pipeline(text, lang=lang)
        text = self.ja_tokenize(text)
    else:
        raise ValueError('It should not reach here')
    if self.do_lowercase_and_remove_accent and not bypass_tokenizer:
        text = lowercase_and_remove_accent(text)
    split_tokens = []
    for token in text:
        if token:
            split_tokens.extend([t for t in self.bpe(token).split(' ')])
    return split_tokens
