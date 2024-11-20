def forward_chunk(self, hidden_states):
    hidden_states = self.decoder(hidden_states)
    return hidden_states
