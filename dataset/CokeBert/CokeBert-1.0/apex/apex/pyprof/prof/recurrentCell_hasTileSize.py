def hasTileSize(name):
    if 'sgemm' in name or '884gemm' in name or 'hgemm' in name:
        return True
    else:
        return False
