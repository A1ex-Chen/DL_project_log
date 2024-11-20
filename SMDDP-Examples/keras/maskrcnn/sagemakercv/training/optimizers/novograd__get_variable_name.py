def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match('^(.*):\\d+$', param_name)
    if m is not None:
        param_name = m.group(1)
        return param_name
