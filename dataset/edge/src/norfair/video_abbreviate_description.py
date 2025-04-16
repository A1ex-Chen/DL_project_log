def abbreviate_description(self, description: str) ->str:
    """Conditionally abbreviate description so that progress bar fits in small terminals"""
    terminal_columns, _ = get_terminal_size()
    space_for_description = int(terminal_columns) - 25
    if len(description) < space_for_description:
        return description
    else:
        return '{} ... {}'.format(description[:space_for_description // 2 -
            3], description[-space_for_description // 2 + 3:])
