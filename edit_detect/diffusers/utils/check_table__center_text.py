def _center_text(text, width):
    text_length = 2 if text == '✅' or text == '❌' else len(text)
    left_indent = (width - text_length) // 2
    right_indent = width - text_length - left_indent
    return ' ' * left_indent + text + ' ' * right_indent
