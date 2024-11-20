def register_conv_template(template: Conversation, override: bool=False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in conv_templates, f'{template.name} has been registered.'
    conv_templates[template.name] = template
