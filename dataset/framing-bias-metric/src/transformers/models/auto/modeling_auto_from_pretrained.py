@classmethod
@replace_list_option_in_docstrings(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING)
@add_start_docstrings(
    'Instantiate one of the model classes of the library---with a multiple choice classification head---from a pretrained model.'
    , AUTO_MODEL_PRETRAINED_DOCSTRING)
def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    """
        Examples::

            >>> from transformers import AutoConfig, AutoModelForNextSentencePrediction

            >>> # Download model and configuration from huggingface.co and cache.
            >>> model = AutoModelForNextSentencePrediction.from_pretrained('bert-base-uncased')

            >>> # Update configuration during loading
            >>> model = AutoModelForNextSentencePrediction.from_pretrained('bert-base-uncased', output_attentions=True)
            >>> model.config.output_attentions
            True

            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            >>> config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            >>> model = AutoModelForNextSentencePrediction.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """
    config = kwargs.pop('config', None)
    if not isinstance(config, PretrainedConfig):
        config, kwargs = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs)
    if type(config) in MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING.keys():
        return MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING[type(config)
            ].from_pretrained(pretrained_model_name_or_path, *model_args,
            config=config, **kwargs)
    raise ValueError(
        """Unrecognized configuration class {} for this kind of AutoModel: {}.
Model type should be one of {}."""
        .format(config.__class__, cls.__name__, ', '.join(c.__name__ for c in
        MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING.keys())))
