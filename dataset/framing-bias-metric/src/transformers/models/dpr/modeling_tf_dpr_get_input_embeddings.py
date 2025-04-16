def get_input_embeddings(self):
    return self.span_predictor.encoder.bert_model.get_input_embeddings()
