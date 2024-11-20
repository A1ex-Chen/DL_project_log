def forward(self, input_ids, input_mask=None, segment_ids=None, labels=None):
    return self.bert(input_ids, input_mask, segment_ids, labels=labels)
