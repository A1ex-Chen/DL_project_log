def forward(self, hidden_states: torch.Tensor, scale: float=1.0
    ) ->torch.Tensor:
    if self.lora_layer is None:
        out = super().forward(hidden_states)
        return out
    else:
        out = super().forward(hidden_states) + scale * self.lora_layer(
            hidden_states)
        return out
