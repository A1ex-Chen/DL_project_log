def __init__(self, num_labels):
    super(HiBERT, self).__init__()
    self.bert = BertForSequenceClassification(BertConfig(num_labels,
        hidden_size=128, num_attention_heads=2, num_hidden_layers=2))
