def load_pretrained(self, state_dict: Mapping[str, Any], strict: bool=True):
    return self.load_state_dict(state_dict, strict)
