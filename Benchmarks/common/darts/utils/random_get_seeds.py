def get_seeds(self):
    return {'PythonHash': self.s.pythonhash, 'PythonRand': self.s.
        pythonrand, 'Numpy': self.s.numpy, 'Torch': self.s.torch}
