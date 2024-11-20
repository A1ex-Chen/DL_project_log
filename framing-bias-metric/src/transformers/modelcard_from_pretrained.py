@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    """
        Instantiate a :class:`~transformers.ModelCard` from a pre-trained model model card.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string, the `model id` of a pretrained model card hosted inside a model repo on huggingface.co.
                  Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under a
                  user or organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a model card file saved using the
                  :func:`~transformers.ModelCard.save_pretrained` method, e.g.: ``./my_model_directory/``.
                - a path or url to a saved model card JSON `file`, e.g.: ``./my_model_directory/modelcard.json``.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model card should be cached if the standard cache
                should not be used.

            kwargs: (`optional`) dict: key/value pairs with which to update the ModelCard object after loading.

                - The values in kwargs of any keys which are model card attributes will be used to override the loaded
                  values.
                - Behavior concerning key/value pairs whose keys are *not* model card attributes is controlled by the
                  `return_unused_kwargs` keyword parameter.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}. The proxies are used on each request.

            find_from_standard_name: (`optional`) boolean, default True:
                If the pretrained_model_name_or_path ends with our standard model or config filenames, replace them
                with our standard modelcard filename. Can be used to directly feed a model/config url and access the
                colocated modelcard.

            return_unused_kwargs: (`optional`) bool:

                - If False, then this function returns just the final model card object.
                - If True, then this functions returns a tuple `(model card, unused_kwargs)` where `unused_kwargs` is a
                  dictionary consisting of the key/value pairs whose keys are not model card attributes: ie the part of
                  kwargs which has not been used to update `ModelCard` and is otherwise ignored.

        Examples::

            modelcard = ModelCard.from_pretrained('bert-base-uncased')    # Download model card from huggingface.co and cache.
            modelcard = ModelCard.from_pretrained('./test/saved_model/')  # E.g. model card was saved using `save_pretrained('./test/saved_model/')`
            modelcard = ModelCard.from_pretrained('./test/saved_model/modelcard.json')
            modelcard = ModelCard.from_pretrained('bert-base-uncased', output_attentions=True, foo=False)

        """
    cache_dir = kwargs.pop('cache_dir', None)
    proxies = kwargs.pop('proxies', None)
    find_from_standard_name = kwargs.pop('find_from_standard_name', True)
    return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)
    if pretrained_model_name_or_path in ALL_PRETRAINED_CONFIG_ARCHIVE_MAP:
        model_card_file = ALL_PRETRAINED_CONFIG_ARCHIVE_MAP[
            pretrained_model_name_or_path]
    elif os.path.isdir(pretrained_model_name_or_path):
        model_card_file = os.path.join(pretrained_model_name_or_path,
            MODEL_CARD_NAME)
    elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(
        pretrained_model_name_or_path):
        model_card_file = pretrained_model_name_or_path
    else:
        model_card_file = hf_bucket_url(pretrained_model_name_or_path,
            filename=MODEL_CARD_NAME, mirror=None)
    if (find_from_standard_name or pretrained_model_name_or_path in
        ALL_PRETRAINED_CONFIG_ARCHIVE_MAP):
        model_card_file = model_card_file.replace(CONFIG_NAME, MODEL_CARD_NAME)
        model_card_file = model_card_file.replace(WEIGHTS_NAME, MODEL_CARD_NAME
            )
        model_card_file = model_card_file.replace(TF2_WEIGHTS_NAME,
            MODEL_CARD_NAME)
    try:
        resolved_model_card_file = cached_path(model_card_file, cache_dir=
            cache_dir, proxies=proxies)
        if resolved_model_card_file == model_card_file:
            logger.info('loading model card file {}'.format(model_card_file))
        else:
            logger.info('loading model card file {} from cache at {}'.
                format(model_card_file, resolved_model_card_file))
        modelcard = cls.from_json_file(resolved_model_card_file)
    except (EnvironmentError, json.JSONDecodeError):
        modelcard = cls()
    to_remove = []
    for key, value in kwargs.items():
        if hasattr(modelcard, key):
            setattr(modelcard, key, value)
            to_remove.append(key)
    for key in to_remove:
        kwargs.pop(key, None)
    logger.info('Model card: %s', str(modelcard))
    if return_unused_kwargs:
        return modelcard, kwargs
    else:
        return modelcard
