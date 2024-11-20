@max_len_single_sentence.setter
def max_len_single_sentence(self, value) ->int:
    if value == self.model_max_length - self.num_special_tokens_to_add(pair
        =False) and self.verbose:
        if not self.deprecation_warnings.get('max_len_single_sentence', False):
            logger.warning(
                "Setting 'max_len_single_sentence' is now deprecated. This value is automatically set up."
                )
        self.deprecation_warnings['max_len_single_sentence'] = True
    else:
        raise ValueError(
            "Setting 'max_len_single_sentence' is now deprecated. This value is automatically set up."
            )
