def from_pretrained(self, load_path):
    state_dict = torch.load(load_path, map_location=self.opt['device'])
    state_dict = align_and_update_state_dicts(self.model.state_dict(),
        state_dict)
    self.model.load_state_dict(state_dict, strict=False)
    return self
