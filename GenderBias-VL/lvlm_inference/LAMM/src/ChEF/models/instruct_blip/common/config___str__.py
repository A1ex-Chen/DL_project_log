def __str__(self):
    s = f'{self.name}={self.val}'
    if self.type is not None:
        s += f', ({self.type})'
    if self.choices is not None:
        s += f', choices: {self.choices}'
    if self.help is not None:
        s += f', ({self.help})'
    return s
