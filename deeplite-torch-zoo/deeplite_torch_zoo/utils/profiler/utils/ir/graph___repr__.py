def __repr__(self):
    text = self.name
    text += ' (' + '\n'
    text += ',\n'.join([('\t' + str(v)) for v in self.inputs]) + '\n'
    text += '):' + '\n'
    text += '\n'.join([('\t' + str(x)) for x in self.nodes]) + '\n'
    text += '\t' + 'return ' + ', '.join([str(v) for v in self.outputs])
    return text
