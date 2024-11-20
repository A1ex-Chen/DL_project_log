def check_inputs(self, prompt, height, width, strength, callback_steps):
    if not isinstance(prompt, str) and not isinstance(prompt, list):
        raise ValueError(
            f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
    if strength < 0 or strength > 1:
        raise ValueError(
            f'The value of strength should in [0.0, 1.0] but is {strength}')
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            f'`height` and `width` have to be divisible by 8 but are {height} and {width}.'
            )
    if callback_steps is None or callback_steps is not None and (not
        isinstance(callback_steps, int) or callback_steps <= 0):
        raise ValueError(
            f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.'
            )
