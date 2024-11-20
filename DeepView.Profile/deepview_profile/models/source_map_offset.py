def offset(self, length):
    return Position(self.line, self.column + length)
