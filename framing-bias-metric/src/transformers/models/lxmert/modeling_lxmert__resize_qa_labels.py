def _resize_qa_labels(self, num_labels):
    cur_qa_logit_layer = self.get_qa_logit_layer()
    new_qa_logit_layer = self._get_resized_qa_labels(cur_qa_logit_layer,
        num_labels)
    self._set_qa_logit_layer(new_qa_logit_layer)
    return self.get_qa_logit_layer()
