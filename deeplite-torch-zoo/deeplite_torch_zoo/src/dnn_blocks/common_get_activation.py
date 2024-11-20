def get_activation(activation_name):
    if activation_name:
        return ACT_TYPE_MAP[activation_name]
    LOGGER.debug(
        'No activation specified for get_activation. Returning nn.Identity()')
    return nn.Identity()
