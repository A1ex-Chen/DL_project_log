def _highlight(code, filename):
    try:
        import pygments
    except ImportError:
        return code
    from pygments.lexers import Python3Lexer, YamlLexer
    from pygments.formatters import Terminal256Formatter
    lexer = Python3Lexer() if filename.endswith('.py') else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style=
        'monokai'))
    return code
