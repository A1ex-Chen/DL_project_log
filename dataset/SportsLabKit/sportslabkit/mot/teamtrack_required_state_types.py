@property
def required_state_types(self):
    motion_model_required_state_types = self.motion_model.required_state_types
    required_state_types = motion_model_required_state_types + ['pred_pt']
    return required_state_types
