def shell(self, target: str=None):
    string = f'{self.compiler} -r '
    if target is None:
        target = self.target
    string += f"-o {target} {' '.join(self.outs)} "
    return string
