def get_conv_template(name: str) ->Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()
