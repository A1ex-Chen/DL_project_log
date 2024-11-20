def forward(self, attn, is_cross: bool, place_in_unet: str):
    key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
    self.step_store[key].append(attn)
