def __init__(self, input_ids, input_mask, segment_ids, input_ent, ent_mask,
    labels):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.labels = labels
    self.input_ent = input_ent
    self.ent_mask = ent_mask
