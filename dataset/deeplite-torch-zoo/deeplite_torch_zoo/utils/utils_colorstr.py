def colorstr(*input):
    """Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')."""
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
