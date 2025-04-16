def populate_model_card(model_card: ModelCard, tags: Union[str, List[str]]=None
    ) ->ModelCard:
    """Populates the `model_card` with library name and optional tags."""
    if model_card.data.library_name is None:
        model_card.data.library_name = 'diffusers'
    if tags is not None:
        if isinstance(tags, str):
            tags = [tags]
        if model_card.data.tags is None:
            model_card.data.tags = []
        for tag in tags:
            model_card.data.tags.append(tag)
    return model_card
