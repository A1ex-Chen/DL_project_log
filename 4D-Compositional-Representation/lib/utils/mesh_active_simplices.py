def active_simplices(self):
    occ = self.values >= self.threshold
    simplices = self.delaunay.simplices
    simplices_occ = occ[simplices]
    active = np.any(simplices_occ, axis=1) & np.any(~simplices_occ, axis=1)
    simplices = self.delaunay.simplices[active]
    return simplices
