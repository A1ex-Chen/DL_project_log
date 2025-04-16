@max_len_sentences_pair.setter
def max_len_sentences_pair(self, value) ->int:
    if value == self.model_max_length - self.num_special_tokens_to_add(pair
        =True) and self.verbose:
        if not self.deprecation_warnings.get('max_len_sentences_pair', False):
            logger.warning(
                "Setting 'max_len_sentences_pair' is now deprecated. This value is automatically set up."
                )
        self.deprecation_warnings['max_len_sentences_pair'] = True
    else:
        raise ValueError(
            "Setting 'max_len_sentences_pair' is now deprecated. This value is automatically set up."
            )
