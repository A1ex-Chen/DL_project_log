def fetch_any_blob(name):
    bb = None
    try:
        bb = workspace.FetchBlob(name)
    except TypeError:
        bb = workspace.FetchInt8Blob(name)
    except Exception as e:
        logger.error('Get blob {} error: {}'.format(name, e))
    return bb
