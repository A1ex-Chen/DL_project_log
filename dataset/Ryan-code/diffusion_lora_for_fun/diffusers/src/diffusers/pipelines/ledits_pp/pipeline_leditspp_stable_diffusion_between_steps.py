def between_steps(self, store_step=True):
    if store_step:
        if self.average:
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i]
        elif len(self.attention_store) == 0:
            self.attention_store = [self.step_store]
        else:
            self.attention_store.append(self.step_store)
        self.cur_step += 1
    self.step_store = self.get_empty_store()
