def tabulate(rows: List[List[Union[str, int]]], headers: List[str]) ->str:
    """
    Inspired by:

    - stackoverflow.com/a/8356620/593036
    - stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    """
    col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
    row_format = ('{{:{}}} ' * len(headers)).format(*col_widths)
    lines = []
    lines.append(row_format.format(*headers))
    lines.append(row_format.format(*[('-' * w) for w in col_widths]))
    for row in rows:
        lines.append(row_format.format(*row))
    return '\n'.join(lines)
