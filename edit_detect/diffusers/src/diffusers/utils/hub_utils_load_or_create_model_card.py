def load_or_create_model_card(repo_id_or_path: str=None, token: Optional[
    str]=None, is_pipeline: bool=False, from_training: bool=False,
    model_description: Optional[str]=None, base_model: str=None, prompt:
    Optional[str]=None, license: Optional[str]=None, widget: Optional[List[
    dict]]=None, inference: Optional[bool]=None) ->ModelCard:
    """
    Loads or creates a model card.

    Args:
        repo_id_or_path (`str`):
            The repo id (e.g., "runwayml/stable-diffusion-v1-5") or local path where to look for the model card.
        token (`str`, *optional*):
            Authentication token. Will default to the stored token. See https://huggingface.co/settings/token for more
            details.
        is_pipeline (`bool`):
            Boolean to indicate if we're adding tag to a [`DiffusionPipeline`].
        from_training: (`bool`): Boolean flag to denote if the model card is being created from a training script.
        model_description (`str`, *optional*): Model description to add to the model card. Helpful when using
            `load_or_create_model_card` from a training script.
        base_model (`str`): Base model identifier (e.g., "stabilityai/stable-diffusion-xl-base-1.0"). Useful
            for DreamBooth-like training.
        prompt (`str`, *optional*): Prompt used for training. Useful for DreamBooth-like training.
        license: (`str`, *optional*): License of the output artifact. Helpful when using
            `load_or_create_model_card` from a training script.
        widget (`List[dict]`, *optional*): Widget to accompany a gallery template.
        inference: (`bool`, optional): Whether to turn on inference widget. Helpful when using
            `load_or_create_model_card` from a training script.
    """
    if not is_jinja_available():
        raise ValueError(
            'Modelcard rendering is based on Jinja templates. Please make sure to have `jinja` installed before using `load_or_create_model_card`. To install it, please run `pip install Jinja2`.'
            )
    try:
        model_card = ModelCard.load(repo_id_or_path, token=token)
    except (EntryNotFoundError, RepositoryNotFoundError):
        if from_training:
            model_card = ModelCard.from_template(card_data=ModelCardData(
                license=license, library_name='diffusers', inference=
                inference, base_model=base_model, instance_prompt=prompt,
                widget=widget), template_path=MODEL_CARD_TEMPLATE_PATH,
                model_description=model_description)
        else:
            card_data = ModelCardData()
            component = 'pipeline' if is_pipeline else 'model'
            if model_description is None:
                model_description = (
                    f'This is the model card of a 🧨 diffusers {component} that has been pushed on the Hub. This model card has been automatically generated.'
                    )
            model_card = ModelCard.from_template(card_data,
                model_description=model_description)
    return model_card
