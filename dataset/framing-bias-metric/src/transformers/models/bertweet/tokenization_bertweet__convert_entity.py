def _convert_entity(match):
    entity_body = match.group(3)
    if match.group(1):
        try:
            if match.group(2):
                number = int(entity_body, 16)
            else:
                number = int(entity_body, 10)
            if 128 <= number <= 159:
                return bytes((number,)).decode('cp1252')
        except ValueError:
            number = None
    elif entity_body in keep:
        return match.group(0)
    else:
        number = html.entities.name2codepoint.get(entity_body)
    if number is not None:
        try:
            return chr(number)
        except (ValueError, OverflowError):
            pass
    return '' if remove_illegal else match.group(0)
