def tie_weights(self):
    """ Tying the encoder and decoders' embeddings together.

       We need for each to get down to the embedding weights. However the
        different model classes are inconsistent to that respect:
        - BertModel: embeddings.word_embeddings
        - RoBERTa: embeddings.word_embeddings
        - XLMModel: embeddings
        - GPT2: wte
        - BertForMaskedLM: bert.embeddings.word_embeddings
        - RobertaForMaskedLM: roberta.embeddings.word_embeddings

        argument of the XEmbedding layer for each model, but it is "blocked"
        by a model-specific keyword (bert, )...
        """
    pass
