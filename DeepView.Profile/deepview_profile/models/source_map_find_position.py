def find_position(self, snippet, line_offset=0):
    for offset, line in enumerate(self._source_by_line[line_offset:]):
        index = line.find(snippet)
        if index == -1:
            continue
        return Position(line_offset + offset, index)
    return None
