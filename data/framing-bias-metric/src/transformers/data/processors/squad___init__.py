def __init__(self, unique_id, start_logits, end_logits, start_top_index=
    None, end_top_index=None, cls_logits=None):
    self.start_logits = start_logits
    self.end_logits = end_logits
    self.unique_id = unique_id
    if start_top_index:
        self.start_top_index = start_top_index
        self.end_top_index = end_top_index
        self.cls_logits = cls_logits
