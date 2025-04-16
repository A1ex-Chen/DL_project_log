def init_weights(self):
    self.span_predictor.encoder.init_weights()
    self.span_predictor.qa_classifier.apply(self.span_predictor.encoder.
        bert_model._init_weights)
    self.span_predictor.qa_outputs.apply(self.span_predictor.encoder.
        bert_model._init_weights)
