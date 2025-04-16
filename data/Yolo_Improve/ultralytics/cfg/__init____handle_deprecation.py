def _handle_deprecation(custom):
    """Handles deprecated configuration keys by mapping them to current equivalents with deprecation warnings."""
    for key in custom.copy().keys():
        if key == 'boxes':
            deprecation_warn(key, 'show_boxes')
            custom['show_boxes'] = custom.pop('boxes')
        if key == 'hide_labels':
            deprecation_warn(key, 'show_labels')
            custom['show_labels'] = custom.pop('hide_labels') == 'False'
        if key == 'hide_conf':
            deprecation_warn(key, 'show_conf')
            custom['show_conf'] = custom.pop('hide_conf') == 'False'
        if key == 'line_thickness':
            deprecation_warn(key, 'line_width')
            custom['line_width'] = custom.pop('line_thickness')
    return custom
