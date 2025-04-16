def format_arguments(self):
    return str([f'{k}' for k in sorted(self.arguments.keys())])
