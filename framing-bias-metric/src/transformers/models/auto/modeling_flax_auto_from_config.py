@classmethod
def from_config(cls, config):
    """
        Instantiates one of the base model classes of the library from a configuration.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                - isInstance of `roberta` configuration class: :class:`~transformers.FlaxRobertaModel` (RoBERTa model)
                - isInstance of `bert` configuration class: :class:`~transformers.FlaxBertModel` (Bert model

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')
            # Download configuration from huggingface.co and cache.
            model = FlaxAutoModel.from_config(config)
            # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
    for config_class, model_class in FLAX_MODEL_MAPPING.items():
        if isinstance(config, config_class):
            return model_class(config)
    raise ValueError(
        f"""Unrecognized configuration class {config.__class__} for this kind of FlaxAutoModel: {cls.__name__}.
Model type should be one of {', '.join(c.__name__ for c in FLAX_MODEL_MAPPING.keys())}."""
        )
