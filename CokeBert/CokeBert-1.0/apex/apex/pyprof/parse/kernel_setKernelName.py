def setKernelName(self, name):
    cadena = demangle(name)
    self.kLongName = cadena
    self.kShortName = getShortName(cadena)
