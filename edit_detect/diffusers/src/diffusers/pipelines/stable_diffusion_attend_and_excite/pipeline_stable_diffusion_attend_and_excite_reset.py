def reset(self):
    self.cur_att_layer = 0
    self.step_store = self.get_empty_store()
    self.attention_store = {}
