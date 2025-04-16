def add_connect_except(self, from_state, to_state, w_group):
    excluded_group_word = [w for w in range(self.vocab_size) if w not in
        w_group]
    self.add_connect(from_state, to_state, excluded_group_word)
