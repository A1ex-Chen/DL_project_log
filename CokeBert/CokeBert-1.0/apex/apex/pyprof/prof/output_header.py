def header(self):
    cadena = ()
    for col in self.cols:
        h = Output.table[col][0]
        cadena = cadena + (h,)
    self.foo(cadena, self.hFormat)
