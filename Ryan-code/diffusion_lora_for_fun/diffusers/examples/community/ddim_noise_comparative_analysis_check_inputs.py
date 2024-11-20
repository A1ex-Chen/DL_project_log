def check_inputs(self, strength):
    if strength < 0 or strength > 1:
        raise ValueError(
            f'The value of strength should in [0.0, 1.0] but is {strength}')
