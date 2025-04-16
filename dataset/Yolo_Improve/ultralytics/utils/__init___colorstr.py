def colorstr(*input):
    """
    Colors a string based on the provided color and style arguments. Utilizes ANSI escape codes.
    See https://en.wikipedia.org/wiki/ANSI_escape_code for more details.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str): A sequence of strings where the first n-1 strings are color and style arguments,
                      and the last string is the one to be colored.

    Supported Colors and Styles:
        Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        Misc: 'end', 'bold', 'underline'

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr("blue", "bold", "hello world")
        >>> "[34m[1mhello world[0m"
    """
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {'black': '\x1b[30m', 'red': '\x1b[31m', 'green': '\x1b[32m',
        'yellow': '\x1b[33m', 'blue': '\x1b[34m', 'magenta': '\x1b[35m',
        'cyan': '\x1b[36m', 'white': '\x1b[37m', 'bright_black': '\x1b[90m',
        'bright_red': '\x1b[91m', 'bright_green': '\x1b[92m',
        'bright_yellow': '\x1b[93m', 'bright_blue': '\x1b[94m',
        'bright_magenta': '\x1b[95m', 'bright_cyan': '\x1b[96m',
        'bright_white': '\x1b[97m', 'end': '\x1b[0m', 'bold': '\x1b[1m',
        'underline': '\x1b[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']
