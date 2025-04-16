def copy(self):
    output = self.output.clone() if self.output is not None else None
    pos = self.pos.clone() if self.pos is not None else None
    return Variable(output, self.name, self.type, pos)
