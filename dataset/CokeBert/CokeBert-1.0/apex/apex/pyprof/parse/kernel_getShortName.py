def getShortName(name):
    """
	Returns a shorter kernel name
	"""
    sname = name.split('<')[0].replace('void ', '').replace('at::', ''
        ).replace('cuda::', '').replace('native::', '').replace(
        '(anonymous namespace)::', '')
    sname = sname.split('(')[0]
    return sname
