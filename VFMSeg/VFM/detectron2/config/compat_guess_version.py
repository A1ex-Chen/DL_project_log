def guess_version(cfg: CN, filename: str) ->int:
    """
    Guess the version of a partial config where the VERSION field is not specified.
    Returns the version, or the latest if cannot make a guess.

    This makes it easier for users to migrate.
    """
    logger = logging.getLogger(__name__)

    def _has(name: str) ->bool:
        cur = cfg
        for n in name.split('.'):
            if n not in cur:
                return False
            cur = cur[n]
        return True
    ret = None
    if _has('MODEL.WEIGHT') or _has('TEST.AUG_ON'):
        ret = 1
    if ret is not None:
        logger.warning("Config '{}' has no VERSION. Assuming it to be v{}."
            .format(filename, ret))
    else:
        ret = _C.VERSION
        logger.warning(
            "Config '{}' has no VERSION. Assuming it to be compatible with latest v{}."
            .format(filename, ret))
    return ret
