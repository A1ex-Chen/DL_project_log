def tc(self):
    if self.op() == 'linear':
        return 1 if '884gemm' in self.name else 0
    else:
        return '-'
