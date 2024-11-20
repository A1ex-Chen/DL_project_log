def forward(self, hidden_states):
    hidden_states = self.conv1(hidden_states)
    hidden_states = self.layer_norm(hidden_states.permute(0, 2, 3, 1)).permute(
        0, 3, 1, 2)
    logits = self.conv2(hidden_states)
    return logits
