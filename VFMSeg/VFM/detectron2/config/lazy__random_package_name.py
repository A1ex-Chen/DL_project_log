def _random_package_name(filename):
    return _CFG_PACKAGE_NAME + str(uuid.uuid4())[:4] + '.' + os.path.basename(
        filename)
