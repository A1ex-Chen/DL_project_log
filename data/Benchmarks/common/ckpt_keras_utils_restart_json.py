def restart_json(gParameters, logger, directory):
    json_file = directory + '/ckpt-info.json'
    if not os.path.exists(json_file):
        msg = 'restart_json(): in: %s model exists but not json!' % directory
        logger.info(msg)
        if not disabled(gParameters, 'require_json'):
            raise Exception(msg)
    with open(json_file) as fp:
        J = json.load(fp)
    logger.debug('ckpt-info.json contains:')
    logger.debug(json.dumps(J, indent=2))
    logger.info('restarting from epoch: %i' % J['epoch'])
    if param(gParameters, 'ckpt_checksum', False, ParamType.BOOLEAN):
        checksum = checksum_file(logger, directory + '/model.h5')
        if checksum != J['checksum']:
            raise Exception('checksum mismatch! directory: ' % directory)
    return J
