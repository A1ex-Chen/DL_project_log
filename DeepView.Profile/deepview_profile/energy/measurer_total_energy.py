def total_energy(self):
    total_energy = 0.0
    for m in self.measurers:
        e = self.measurers[m].total_energy()
        if e is not None:
            total_energy += e
    return total_energy
