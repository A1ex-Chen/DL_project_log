def forward(self, hidden_states):
    for block in self.blocks:
        hidden_states = block(hidden_states)
    return hidden_states
