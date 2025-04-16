def __init__(self, input_ids, attention_mask, token_type_ids, label):
    self.input_ids = input_ids
    self.attention_mask = attention_mask
    self.token_type_ids = token_type_ids
    self.label = label
