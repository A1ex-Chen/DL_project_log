def seqcompare(elem):
    """
			Sorting function for sequence markers
			"""
    assert ', seq = ' in elem
    l = elem.split(' = ')
    return l[1] + l[0]
