def tc(self):
    if 'gemm' in self.name:
        return 1 if '884gemm' in self.name else 0
    else:
        return '-'
