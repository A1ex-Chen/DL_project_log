@staticmethod
def clean_up_tokenization(out_string):
    """ Clean up a list of simple English tokenization artifacts like spaces before punctuations and abreviated forms.
        """
    out_string = out_string.replace(' .', '.').replace(' ?', '?').replace(' !',
        '!').replace(' ,', ',').replace(" ' ", "'").replace(" n't", "n't"
        ).replace(" 'm", "'m").replace(' do not', " don't").replace(" 's", "'s"
        ).replace(" 've", "'ve").replace(" 're", "'re")
    return out_string
