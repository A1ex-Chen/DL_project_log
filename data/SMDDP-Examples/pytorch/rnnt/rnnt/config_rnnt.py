def rnnt(conf):
    return validate_and_fill(RNNT, conf['rnnt'], optional=['n_classes'])
