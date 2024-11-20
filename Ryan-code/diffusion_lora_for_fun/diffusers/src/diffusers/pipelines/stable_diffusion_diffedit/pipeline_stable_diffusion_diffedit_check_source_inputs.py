def check_source_inputs(self, source_prompt=None, source_negative_prompt=
    None, source_prompt_embeds=None, source_negative_prompt_embeds=None):
    if source_prompt is not None and source_prompt_embeds is not None:
        raise ValueError(
            f'Cannot forward both `source_prompt`: {source_prompt} and `source_prompt_embeds`: {source_prompt_embeds}.  Please make sure to only forward one of the two.'
            )
    elif source_prompt is None and source_prompt_embeds is None:
        raise ValueError(
            'Provide either `source_image` or `source_prompt_embeds`. Cannot leave all both of the arguments undefined.'
            )
    elif source_prompt is not None and (not isinstance(source_prompt, str) and
        not isinstance(source_prompt, list)):
        raise ValueError(
            f'`source_prompt` has to be of type `str` or `list` but is {type(source_prompt)}'
            )
    if (source_negative_prompt is not None and 
        source_negative_prompt_embeds is not None):
        raise ValueError(
            f'Cannot forward both `source_negative_prompt`: {source_negative_prompt} and `source_negative_prompt_embeds`: {source_negative_prompt_embeds}. Please make sure to only forward one of the two.'
            )
    if (source_prompt_embeds is not None and source_negative_prompt_embeds
         is not None):
        if source_prompt_embeds.shape != source_negative_prompt_embeds.shape:
            raise ValueError(
                f'`source_prompt_embeds` and `source_negative_prompt_embeds` must have the same shape when passed directly, but got: `source_prompt_embeds` {source_prompt_embeds.shape} != `source_negative_prompt_embeds` {source_negative_prompt_embeds.shape}.'
                )
