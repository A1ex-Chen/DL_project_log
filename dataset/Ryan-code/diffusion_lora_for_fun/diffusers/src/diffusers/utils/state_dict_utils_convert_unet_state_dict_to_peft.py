def convert_unet_state_dict_to_peft(state_dict):
    """
    Converts a state dict from UNet format to diffusers format - i.e. by removing some keys
    """
    mapping = UNET_TO_DIFFUSERS
    return convert_state_dict(state_dict, mapping)
