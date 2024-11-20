def _ensure_lines(lines):
    if isinstance(lines, str):
        lines = lines.splitlines(False)
    return lines
