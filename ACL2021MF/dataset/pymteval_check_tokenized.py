def check_tokenized(self, pred_sent, ref_sents):
    """Tokenize the predicted sentence and reference sentences, if they are not tokenized.
        @param pred_sent: system output / predicted sentence
        @param ref_sent: a list of corresponding reference sentences
        @return: a tuple of (pred_sent, ref_sent) where everything is tokenized
        """
    pred_sent = pred_sent if isinstance(pred_sent, list) else self.tokenize(
        pred_sent)
    ref_sents = [(ref_sent if isinstance(ref_sent, list) else self.tokenize
        (ref_sent)) for ref_sent in ref_sents]
    return pred_sent, ref_sents
