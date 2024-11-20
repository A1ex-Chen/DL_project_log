def can_generate(self) ->bool:
    """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
    if 'GenerationMixin' in str(self.prepare_inputs_for_generation.__func__):
        return False
    return True
