def _should_continue(line, indent):
    return line.startswith(indent) or len(line) <= 1 or re.search(
        '^\\s*\\)(\\s*->.*:|:)\\s*$', line) is not None
