@staticmethod
def clean_up_tokenization(out_string: str) ->str:
    """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

        Args:
            out_string (:obj:`str`): The text to clean up.

        Returns:
            :obj:`str`: The cleaned-up string.
        """
    out_string = out_string.replace(' .', '.').replace(' ?', '?').replace(' !',
        '!').replace(' ,', ',').replace(" ' ", "'").replace(" n't", "n't"
        ).replace(" 'm", "'m").replace(" 's", "'s").replace(" 've", "'ve"
        ).replace(" 're", "'re")
    return out_string
