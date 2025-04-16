def __repr__(self):
    text = ', '.join([str(v) for v in self.outputs])
    text += ' = ' + self.operator
    if self.attributes:
        text += '[' + ', '.join([(str(k) + ' = ' + str(v)) for k, v in self
            .attributes.items()]) + ']'
    text += '(' + ', '.join([str(v) for v in self.inputs]) + ')'
    return text
