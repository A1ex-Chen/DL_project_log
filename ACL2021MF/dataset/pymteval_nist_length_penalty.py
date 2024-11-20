def nist_length_penalty(self, lsys, avg_lref):
    """Compute the NIST length penalty, based on system output length & average reference length.
        @param lsys: total system output length
        @param avg_lref: total average reference length
        @return: NIST length penalty term
        """
    ratio = lsys / float(avg_lref)
    if ratio >= 1:
        return 1
    if ratio <= 0:
        return 0
    return math.exp(-self.BETA * math.log(ratio) ** 2)
