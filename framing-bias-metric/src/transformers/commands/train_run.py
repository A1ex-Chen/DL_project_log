def run(self):
    if self.framework == 'tf':
        return self.run_tf()
    return self.run_torch()
