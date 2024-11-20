@register_to_config
def __init__(self, mask_token_id: int, masking_schedule: str='cosine'):
    self.temperatures = None
    self.timesteps = None
