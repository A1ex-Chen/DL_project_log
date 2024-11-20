def forward(self, hidden_states, num_frames=1):
    hidden_states = hidden_states[None, :].reshape((-1, num_frames) +
        hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
    identity = hidden_states
    hidden_states = self.conv1(hidden_states)
    hidden_states = self.conv2(hidden_states)
    hidden_states = self.conv3(hidden_states)
    hidden_states = self.conv4(hidden_states)
    hidden_states = identity + hidden_states
    hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape((
        hidden_states.shape[0] * hidden_states.shape[2], -1) +
        hidden_states.shape[3:])
    return hidden_states
