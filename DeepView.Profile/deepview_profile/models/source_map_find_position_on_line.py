def find_position_on_line(self, snippet, offset_position):
    if offset_position.line >= len(self._source_by_line):
        return None
    index = self._source_by_line[offset_position.line].find(snippet,
        offset_position.column)
    if index == -1:
        return None
    else:
        return Position(offset_position.line, index)
