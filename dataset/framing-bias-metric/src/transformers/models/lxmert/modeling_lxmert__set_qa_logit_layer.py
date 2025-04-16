def _set_qa_logit_layer(self, qa_logit_layer):
    self.answer_head.logit_fc[-1] = qa_logit_layer
