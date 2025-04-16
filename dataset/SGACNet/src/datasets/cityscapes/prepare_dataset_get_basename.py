def get_basename(fp):
    return '_'.join(os.path.basename(fp).split('_')[:3])
