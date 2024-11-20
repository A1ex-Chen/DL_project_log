@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
    """ Instantiate a one of the tokenizer classes of the library
        from a pre-trained model vocabulary.

        The tokenizer class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `camembert`: CamembertTokenizer (CamemBERT model)
            - contains `distilbert`: DistilBertTokenizer (DistilBert model)
            - contains `roberta`: RobertaTokenizer (RoBERTa model)
            - contains `bert`: BertTokenizer (Bert model)
            - contains `openai-gpt`: OpenAIGPTTokenizer (OpenAI GPT model)
            - contains `gpt2`: GPT2Tokenizer (OpenAI GPT-2 model)
            - contains `ctrl`: CTRLTokenizer (Salesforce CTRL model)
            - contains `transfo-xl`: TransfoXLTokenizer (Transformer-XL model)
            - contains `xlnet`: XLNetTokenizer (XLNet model)
            - contains `xlm`: XLMTokenizer (XLM model)

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a predefined tokenizer to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing vocabulary files required by the tokenizer, for instance saved using the :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g.: ``./my_model_directory/``.
                - (not applicable to all derived classes) a path or url to a single saved vocabulary file if and only if the tokenizer only requires a single vocabulary file (e.g. Bert, XLNet), e.g.: ``./my_model_directory/vocab.txt``.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the vocabulary files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            inputs: (`optional`) positional arguments: will be passed to the Tokenizer ``__init__`` method.

            kwargs: (`optional`) keyword arguments: will be passed to the Tokenizer ``__init__`` method. Can be used to set special tokens like ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``, ``additional_special_tokens``. See parameters in the doc string of :class:`~transformers.PreTrainedTokenizer` for details.

        Examples::

            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')    # Download vocabulary from S3 and cache.
            tokenizer = AutoTokenizer.from_pretrained('./test/bert_saved_model/')  # E.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`

        """
    if 'distilbert' in pretrained_model_name_or_path:
        return DistilBertTokenizer.from_pretrained(
            pretrained_model_name_or_path, *inputs, **kwargs)
    elif 'camembert' in pretrained_model_name_or_path:
        return CamembertTokenizer.from_pretrained(pretrained_model_name_or_path
            , *inputs, **kwargs)
    elif 'roberta' in pretrained_model_name_or_path:
        return RobertaTokenizer.from_pretrained(pretrained_model_name_or_path,
            *inputs, **kwargs)
    elif 'bert' in pretrained_model_name_or_path:
        return BertTokenizer.from_pretrained(pretrained_model_name_or_path,
            *inputs, **kwargs)
    elif 'openai-gpt' in pretrained_model_name_or_path:
        return OpenAIGPTTokenizer.from_pretrained(pretrained_model_name_or_path
            , *inputs, **kwargs)
    elif 'gpt2' in pretrained_model_name_or_path:
        return GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path,
            *inputs, **kwargs)
    elif 'transfo-xl' in pretrained_model_name_or_path:
        return TransfoXLTokenizer.from_pretrained(pretrained_model_name_or_path
            , *inputs, **kwargs)
    elif 'xlnet' in pretrained_model_name_or_path:
        return XLNetTokenizer.from_pretrained(pretrained_model_name_or_path,
            *inputs, **kwargs)
    elif 'xlm' in pretrained_model_name_or_path:
        return XLMTokenizer.from_pretrained(pretrained_model_name_or_path,
            *inputs, **kwargs)
    elif 'ctrl' in pretrained_model_name_or_path:
        return CTRLTokenizer.from_pretrained(pretrained_model_name_or_path,
            *inputs, **kwargs)
    raise ValueError(
        "Unrecognized model identifier in {}. Should contains one of 'bert', 'openai-gpt', 'gpt2', 'transfo-xl', 'xlnet', 'xlm', 'roberta', 'camembert', 'ctrl'"
        .format(pretrained_model_name_or_path))
