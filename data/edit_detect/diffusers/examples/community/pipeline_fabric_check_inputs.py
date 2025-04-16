def check_inputs(self, prompt, negative_prompt=None, liked=None, disliked=
    None, height=None, width=None):
    if prompt is None:
        raise ValueError(
            'Provide `prompt`. Cannot leave both `prompt` undefined.')
    elif prompt is not None and (not isinstance(prompt, str) and not
        isinstance(prompt, list)):
        raise ValueError(
            f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
    if negative_prompt is not None and (not isinstance(negative_prompt, str
        ) and not isinstance(negative_prompt, list)):
        raise ValueError(
            f'`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}'
            )
    if liked is not None and not isinstance(liked, list):
        raise ValueError(
            f'`liked` has to be of type `list` but is {type(liked)}')
    if disliked is not None and not isinstance(disliked, list):
        raise ValueError(
            f'`disliked` has to be of type `list` but is {type(disliked)}')
    if height is not None and not isinstance(height, int):
        raise ValueError(
            f'`height` has to be of type `int` but is {type(height)}')
    if width is not None and not isinstance(width, int):
        raise ValueError(
            f'`width` has to be of type `int` but is {type(width)}')
